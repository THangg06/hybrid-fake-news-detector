import os
import sys
import json
import re
from pathlib import Path
from django.conf import settings
from django.utils import timezone

# Thêm path để import từ src folder
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

class FakeNewsPredictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return

        self._deps_loaded = False
        self._np = None
        self._joblib = None
        self._torch = None
        self._RobertaTokenizer = None
        self._RobertaModel = None
        self._clean_text_for_tfidf = None
        self._clean_text_for_roberta = None
        self._combine_features = None
        self._extra_features = None

        self.device = None
        self.models_ready = False
        self.review_confidence_threshold = 0.60
        self.consistency_similarity_threshold = 92.0
        self.max_guardrail_shift = 0.10
        self._load_models()
        self._initialized = True

    def _load_dependencies(self):
        """Load optional ML dependencies without crashing Django startup."""
        if self._deps_loaded:
            return

        import numpy as np
        import joblib
        import torch
        from transformers import RobertaTokenizer, RobertaModel
        from preprocessing import clean_text_for_tfidf, clean_text_for_roberta
        from feature_combiner import combine_features, extra_features

        self._np = np
        self._joblib = joblib
        self._torch = torch
        self._RobertaTokenizer = RobertaTokenizer
        self._RobertaModel = RobertaModel
        self._clean_text_for_tfidf = clean_text_for_tfidf
        self._clean_text_for_roberta = clean_text_for_roberta
        self._combine_features = combine_features
        self._extra_features = extra_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._deps_loaded = True
    
    def _load_models(self):
        """Load tất cả models cần thiết"""
        # Đường dẫn đến models (root folder của project)
        # settings.BASE_DIR = django_app folder
        # Lên 1 cấp để tới level chứa src/ folder
        base_path = Path(settings.BASE_DIR).parent
        
        print(f"[DEBUG] Base path: {base_path}")
        print(f"[DEBUG] Base path exists: {base_path.exists()}")
        
        try:
            self._load_dependencies()

            # Load XGBoost model
            fake_model_path = base_path / "fake_news_xgboost.pkl"
            tfidf_path = base_path / "tfidf_vectorizer.pkl"
            
            print(f"[DEBUG] Loading: {fake_model_path}")
            print(f"[DEBUG] File exists: {fake_model_path.exists()}")
            
            self.clf_model = self._joblib.load(str(fake_model_path))
            
            # Load TF-IDF vectorizer
            self.tfidf_vectorizer = self._joblib.load(str(tfidf_path))

            threshold_path = base_path / "rumor_threshold.json"
            self.rumor_threshold = 0.5
            self.best_iteration = None
            if threshold_path.exists():
                with open(threshold_path, "r", encoding="utf-8") as f:
                    threshold_payload = json.load(f)
                    self.rumor_threshold = float(threshold_payload.get("rumor_threshold", 0.5))
                    best_iter_val = threshold_payload.get("best_iteration")
                    if best_iter_val is not None:
                        self.best_iteration = int(best_iter_val)
            
            # Load RoBERTa
            self.tokenizer = self._RobertaTokenizer.from_pretrained("roberta-base")
            self.roberta_model = self._RobertaModel.from_pretrained("roberta-base")
            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            
            self.models_ready = True
            print("✓ Tất cả models đã được load thành công")
            
        except Exception as e:
            self.models_ready = False
            print(f"⚠ Cảnh báo: Models chưa sẵn sàng.")
            print(f"   Chi tiết lỗi: {str(e)}")
            import traceback
            traceback.print_exc()
            print("   Django vẫn có thể chạy, nhưng dự đoán sẽ không hoạt động cho đến khi models được cấu hình.")
    
    def get_roberta_embedding(self, text):
        """Lấy RoBERTa embedding cho text (dùng [CLS] token)"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with self._torch.no_grad():
            outputs = self.roberta_model(**inputs)
        
        # ✅ Use [CLS] token (index 0) for classification
        embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding.cpu().numpy().flatten()

    def _apply_rumor_language_guardrail(self, title, truth_prob, rumor_prob):
        """Guardrail mềm: chỉ hiệu chỉnh nhẹ ở vùng sát ngưỡng, không ép nhãn cứng."""
        text = (title or "").lower()

        strong_cues = [
            # Y tế - Medical conspiracy/claims
            "miracle treatment",
            "miracle cure",
            "cure instantly",
            "completely cure",
            "100% effective",
            "secret vaccine",
            "already prevents all",
            "kills all cancer",
            "instant healing",
            "cure all diseases",
            "hidden side effects",
            "forbidden treatment",
            "banned by government",
            "suppressed cure",
            "big pharma conspiracy",
            
            # Chính trị - Political conspiracy
            "hidden agenda",
            "control the education system",
            "secretly announced",
            "covert plan",
            "hidden government",
            "exposed conspiracy",
            "leaked documents",
            "cover up",
            "secretly controlling",
            "shadow government",
            "rigged election",
            "fake news media",
            
            # Kinh tế - Financial/business claims
            "hidden product sales",
            "secret formula",
            "make money fast",
            "guaranteed profit",
            "exclusive deal",
            "limited time only",
            "secret billionaire method",
            "hidden riches",
            "banned technique",
            
            # Công nghệ - Tech breakthrough
            "revolutionary invention",
            "breakthrough discovery",
            "never seen before",
            "secret technology",
            "hidden AI",
            "suppressed innovation",
            "banned technology",
            "impossible invention",
            
            # Xã hội - Social claims
            "shocking truth",
            "finally revealed",
            "forbidden knowledge",
            "banned information",
            "suppressed evidence",
            "uncovered scandal",
            "exposed secret",
        ]
        weak_cues = [
            # Từ nhấn mạnh
            "secretly",
            "hidden",
            "miracle",
            "instantly",
            "completely",
            "absolutely",
            "definitely",
            "certainly",
            "undoubtedly",
            "incredible",
            "shocking",
            "unbelievable",
            "devastating",
            "amazing",
            "extraordinary",
            "unimaginable",
            
            # Tính từ quá mức
            "all diseases",
            "can cure",
            "secret",
            "forbidden",
            "banned",
            "exposed",
            "revealed",
            "suppressed",
            "covered up",
            "leaked",
            "unknown",
            "mysterious",
            "conspiracy",
            "scandal",
            
            # Hành động ẩn
            "covert",
            "covertly",
            "hidden away",
            "suppressed",
            "censored",
            "silenced",
            "wiped out",
            "erased",
            "deleted",
            
            # Tính khẩn cấp
            "urgent",
            "critical",
            "emergency",
            "crisis",
            "catastrophe",
            "disaster",
            "apocalypse",
            "day of reckoning",
        ]
        hedge_cues = [
            # Modal verbs - giảm tuyệt đối
            "may",
            "could",
            "might",
            "might be",
            "appears to",
            "seems to",
            "possibly",
            "probably",
            "likely",
            
            # Từ chỉ không chắc chắn
            "according to",
            "study",
            "testing",
            "preliminary",
            "initial",
            "early",
            "suggests",
            "indicates",
            "shows",
            "research indicates",
            "study shows",
            
            # Kế hoạch và ý định (chưa thực hiện)
            "plan to improve",
            "plan to",
            "aimed to",
            "intended to",
            "proposed to",
            "designed to",
            "hoping to",
            "trying to",
            
            # Cộng đồng/vẫn đang nghiên cứu
            "community",
            "some people",
            "many believe",
            "allegedly",
            "reportedly",
            "anecdotal",
            "anecdotally",
            "unconfirmed",
            "unverified",
        ]

        score = 0
        matched = []

        for cue in strong_cues:
            if cue in text:
                score += 2
                matched.append(cue)

        for cue in weak_cues:
            if cue in text:
                score += 1
                matched.append(cue)

        for cue in hedge_cues:
            if cue in text:
                score -= 1

        # Cụm tuyệt đối hóa trong ngữ cảnh y tế là tín hiệu rumor mạnh.
        if re.search(r"\b(cure|prevents?)\b.*\b(instantly|completely|all)\b", text):
            score += 2
            matched.append("absolute-medical-claim")

        adjusted_truth = float(truth_prob)
        adjusted_rumor = float(rumor_prob)
        near_boundary = abs(adjusted_rumor - adjusted_truth) <= 0.25
        shift = min(max(score, 0) * 0.02, self.max_guardrail_shift)

        # Chỉ dịch chuyển nhẹ khi model đang lưỡng lự.
        if score >= 2 and near_boundary:
            adjusted_rumor = min(0.9999, adjusted_rumor + shift)
            adjusted_truth = max(0.0001, 1.0 - adjusted_rumor)

        review_reason = ""
        if score >= 4 and not near_boundary:
            review_reason = (
                "Ngôn ngữ có dấu hiệu giật gân/tuyệt đối cao nhưng model khá chắc. "
                "Nên kiểm tra thủ công để tránh overrule bằng keyword."
            )

        return adjusted_truth, adjusted_rumor, {
            'score': score,
            'matched_cues': matched[:8],
            'applied': score >= 2 and near_boundary,
            'near_boundary': near_boundary,
            'shift': round(shift, 4),
            'review_reason': review_reason,
        }

    def _predict_from_embedding(self, title, clean_tfidf, roberta_embedding):
        """Chạy model phân loại với embedding đã được tạo sẵn."""
        tfidf_features = self.tfidf_vectorizer.transform([clean_tfidf]).toarray()
        roberta_features = self._np.array([roberta_embedding])

        import pandas as pd
        content_df = pd.DataFrame({'content': [title]})
        engineered = self._extra_features(content_df)

        X = self._combine_features(tfidf_features, roberta_features, engineered_features=engineered)

        if self.best_iteration is not None and self.best_iteration >= 0:
            probs = self.clf_model.predict_proba(X, iteration_range=(0, self.best_iteration + 1))[0]
        else:
            probs = self.clf_model.predict_proba(X)[0]

        pred_model = int(probs[1] >= self.rumor_threshold)
        truth_prob = float(probs[0])
        rumor_prob = float(probs[1])

        truth_prob, rumor_prob, guardrail_info = self._apply_rumor_language_guardrail(
            title=title,
            truth_prob=truth_prob,
            rumor_prob=rumor_prob,
        )
        pred_model = int(rumor_prob >= self.rumor_threshold)
        pred_db = 0 if pred_model == 1 else 1

        label = "Fake" if pred_db == 0 else "Real"
        confidence = (rumor_prob if pred_db == 0 else truth_prob) * 100
        max_prob = max(truth_prob, rumor_prob)
        review_needed = max_prob < self.review_confidence_threshold

        return {
            'prediction': int(pred_db),
            'label': label,
            'truth_prob': round(truth_prob, 4),
            'rumor_prob': round(rumor_prob, 4),
            'real_prob': round(truth_prob, 4),
            'fake_prob': round(rumor_prob, 4),
            'confidence': round(confidence, 2),
            'language_guardrail': guardrail_info,
            'review_needed': bool(review_needed or bool(guardrail_info.get('review_reason'))),
            'review_reason': (
                f"Model confidence thấp ({round(max_prob * 100, 2)}%), cần kiểm tra thủ công."
                if review_needed else guardrail_info.get('review_reason', "")
            )
        }

    def find_similar_articles(self, query_embedding, top_k=3, time_scope='before', reference_time=None):
        """Tìm top-k bài viết tương tự bằng cosine similarity, có filter theo mốc thời gian."""
        from .models import SemanticEmbedding

        ref_time = reference_time or timezone.now()
        queryset = SemanticEmbedding.objects.select_related('article').all()

        if time_scope == 'before':
            queryset = queryset.filter(article__created_at__lte=ref_time)
        elif time_scope == 'after':
            queryset = queryset.filter(article__created_at__gte=ref_time)
        elif time_scope == 'all':
            pass
        else:
            raise ValueError("time_scope phải là 'before', 'after' hoặc 'all'.")

        rows = list(queryset)
        if not rows:
            return []

        matrix = []
        metadata = []
        query = self._np.asarray(query_embedding, dtype=self._np.float32)
        query_norm = float(self._np.linalg.norm(query))
        if query_norm == 0:
            return []

        # Chuẩn hóa title để tránh trả về các bản ghi gần như trùng nội dung.
        def _normalize_title(text):
            norm = (text or "").lower()
            norm = re.sub(r"[^a-z0-9\s]", " ", norm)
            norm = re.sub(r"\s+", " ", norm).strip()
            return norm

        query_title_norm = None
        if hasattr(self, "_last_query_title"):
            query_title_norm = _normalize_title(self._last_query_title)

        for row in rows:
            vec = self._np.asarray(row.embedding, dtype=self._np.float32)
            if vec.ndim != 1 or vec.shape[0] != query.shape[0]:
                continue
            if query_title_norm:
                candidate_title_norm = _normalize_title(row.article.title)
                if candidate_title_norm == query_title_norm:
                    continue
            vec_norm = float(self._np.linalg.norm(vec))
            if vec_norm == 0:
                continue
            matrix.append(vec)
            metadata.append(row)

        if not matrix:
            return []

        emb_matrix = self._np.vstack(matrix)

        # Mean-centering giúp giảm hiện tượng cosine dồn lên 99.xx giữa các câu cùng domain.
        mean_vec = emb_matrix.mean(axis=0)
        emb_centered = emb_matrix - mean_vec
        query_centered = query - mean_vec

        matrix_norm = self._np.linalg.norm(emb_centered, axis=1)
        query_centered_norm = float(self._np.linalg.norm(query_centered))
        sims = emb_centered.dot(query_centered) / (matrix_norm * query_centered_norm + 1e-12)

        sorted_indices = self._np.argsort(-sims)

        similar_articles = []
        for idx in sorted_indices:
            row = metadata[int(idx)]
            article = row.article
            sim_value = float(sims[int(idx)])

            # Bỏ qua near-duplicate cực cao để top-k đa dạng hơn.
            if sim_value >= 0.995:
                continue

            similar_articles.append({
                'article_id': article.id,
                'title': article.title,
                'prediction': article.prediction,
                'prediction_label': article.get_prediction_label(),
                'similarity': round(sim_value * 100, 2),
                'created_at': article.created_at.isoformat()
            })
            if len(similar_articles) >= top_k:
                break

        # Fallback: nếu lọc duplicate quá mạnh khiến thiếu kết quả thì lấy lại top chưa lọc.
        if len(similar_articles) < top_k:
            used_ids = {x['article_id'] for x in similar_articles}
            for idx in sorted_indices:
                row = metadata[int(idx)]
                article = row.article
                if article.id in used_ids:
                    continue
                similar_articles.append({
                    'article_id': article.id,
                    'title': article.title,
                    'prediction': article.prediction,
                    'prediction_label': article.get_prediction_label(),
                    'similarity': round(float(sims[int(idx)]) * 100, 2),
                    'created_at': article.created_at.isoformat()
                })
                if len(similar_articles) >= top_k:
                    break

        return similar_articles

    def predict_with_similarity(self, title, top_k=3, time_scope='before', reference_time=None):
        """
        Pipeline đầy đủ:
        1) Tạo embedding cho bài mới
        2) Tìm top-k bài tương tự
        3) Dự đoán bằng TF-IDF + RoBERTa + XGBoost
        """
        if not self.models_ready:
            raise RuntimeError("Models chưa được tải. Vui lòng chạy training script trước.")

        clean_tfidf = self._clean_text_for_tfidf(title)
        clean_roberta = self._clean_text_for_roberta(title)
        self._last_query_title = title
        query_embedding = self.get_roberta_embedding(clean_roberta)

        result = self._predict_from_embedding(
            title=title,
            clean_tfidf=clean_tfidf,
            roberta_embedding=query_embedding
        )
        similar_articles = self.find_similar_articles(
            query_embedding,
            top_k=top_k,
            time_scope=time_scope,
            reference_time=reference_time,
        )
        result['similar_articles'] = similar_articles
        result['time_scope'] = time_scope

        # Nếu bài gần nhất quá giống nhưng nhãn ngược, ép trạng thái review để tránh kết luận quá cứng.
        if similar_articles:
            top_similar = similar_articles[0]
            top_prediction = top_similar.get('prediction')
            top_similarity = float(top_similar.get('similarity', 0.0))
            if (
                top_prediction in (0, 1)
                and top_similarity >= self.consistency_similarity_threshold
                and int(top_prediction) != int(result['prediction'])
            ):
                result['review_needed'] = True
                result['review_reason'] = (
                    f"Bài gần nhất có độ tương đồng {top_similarity}% nhưng nhãn ngược. "
                    "Nên kiểm tra thủ công trước khi kết luận."
                )
                result['consistency_warning'] = {
                    'similarity': top_similarity,
                    'reference_article_id': top_similar.get('article_id'),
                    'reference_prediction': top_similar.get('prediction'),
                    'reference_prediction_label': top_similar.get('prediction_label'),
                }
            else:
                result['consistency_warning'] = None
        else:
            result['consistency_warning'] = None

        result['_embedding_vector'] = query_embedding
        return result

    def save_article_embedding(self, article, embedding_vector):
        """Lưu embedding của bài viết vào database."""
        from .models import SemanticEmbedding

        vector = self._np.asarray(embedding_vector, dtype=self._np.float32)
        payload = vector.astype(float).tolist()

        SemanticEmbedding.objects.update_or_create(
            article=article,
            defaults={
                'embedding': payload,
                'embedding_dim': int(vector.shape[0])
            }
        )
    
    def predict(self, title):
        """
        Dự đoán rumor/truth dựa trên text
        
        Args:
            title (str): Tiêu đề bài viết
        
        Returns:
            dict: {
                'prediction': 0 hoặc 1 (Django convention: 0=Fake, 1=Real),
                'label': 'Fake' hoặc 'Real',
                'truth_prob': float,
                'rumor_prob': float,
                'confidence': float
            }
        """
        try:
            result = self.predict_with_similarity(title=title, top_k=3)
            result.pop('_embedding_vector', None)
            return result
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán: {e}")
            import traceback
            traceback.print_exc()
            raise


# Singleton instance
predictor = None

def get_predictor():
    """Lấy singleton predictor instance"""
    global predictor
    if predictor is None:
        predictor = FakeNewsPredictor()
    return predictor
