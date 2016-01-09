# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='InitialCondition',
            fields=[
                ('id', models.AutoField(auto_created=True, verbose_name='ID', serialize=False, primary_key=True)),
                ('title', models.CharField(max_length=400)),
                ('data', models.TextField()),
                ('length', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='SimResult',
            fields=[
                ('id', models.AutoField(auto_created=True, verbose_name='ID', serialize=False, primary_key=True)),
                ('V', models.CharField(max_length=10)),
                ('R', models.IntegerField()),
                ('mode', models.BooleanField()),
                ('T', models.IntegerField()),
                ('location', models.CharField(max_length=500)),
                ('completed', models.BooleanField()),
                ('IC', models.ForeignKey(to='qcaadmin.InitialCondition')),
            ],
        ),
    ]
