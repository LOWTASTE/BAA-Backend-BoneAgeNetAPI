from django.db import models


# Create your models here.

class BAAModel(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=100, blank=True, default='')
    epoch = models.IntegerField(default=0)
    accuracy_rate = models.DecimalField(default=0.0, decimal_places=4, max_digits=10)
    algorithm = models.ForeignKey('Algorithm', on_delete=models.CASCADE)
    path = models.CharField(max_length=250, blank=False, default='')
    description = models.TextField(default='')

    class Meta:
        ordering = ('created',)


class BAAModel_Data(models.Model):
    model = models.ForeignKey('BAAModel', on_delete=models.CASCADE)
    pic_url = models.URLField(default="")


class Algorithm(models.Model):
    name = models.CharField(max_length=100, blank=True, default='')
