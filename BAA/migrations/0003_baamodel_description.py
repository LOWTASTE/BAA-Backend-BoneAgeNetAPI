# Generated by Django 3.2.5 on 2022-04-12 02:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('BAA', '0002_baamodel_path'),
    ]

    operations = [
        migrations.AddField(
            model_name='baamodel',
            name='description',
            field=models.TextField(default=''),
        ),
    ]
