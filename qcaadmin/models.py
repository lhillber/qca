from django.db import models

class InitialCondition(models.Model):
    title = models.CharField(max_length=400)
    data = models.TextField()
    length = models.IntegerField()

class SimResult(models.Model):
    V = models.CharField(max_length=10) #local gate
    R = models.IntegerField() #ECA Rule: 6, 102, 150, etc
    # possibilities: 204, 201, 198, 195, 156, 153, 150, 147, 108, 105, 102, 99, 60, 57, 54, 51
    # interesting:
    IC = models.ForeignKey(InitialCondition)
    mode = models.BooleanField() #True: Sweep, False: Block
    T = models.IntegerField()
    location = models.CharField(max_length=500)
    completed = models.BooleanField()


