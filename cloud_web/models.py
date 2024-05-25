from django.db import models


class Developer(models.Model):
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=100)
    bio = models.TextField()


class ApplicationFeature(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()

