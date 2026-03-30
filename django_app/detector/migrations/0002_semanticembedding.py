from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('detector', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='SemanticEmbedding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', models.JSONField(verbose_name='RoBERTa embedding')),
                ('embedding_dim', models.IntegerField(default=768, verbose_name='Số chiều embedding')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='Cập nhật')),
                ('article', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='semantic_embedding', to='detector.newsarticle')),
            ],
            options={
                'verbose_name': 'Embedding ngữ nghĩa',
                'verbose_name_plural': 'Embeddings ngữ nghĩa',
            },
        ),
    ]
