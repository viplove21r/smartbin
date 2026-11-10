from django.contrib import admin
from .models import WasteScan, CompostItem, CompostPit, CompostPitItem

admin.site.register(WasteScan)
admin.site.register(CompostItem)
admin.site.register(CompostPit)
admin.site.register(CompostPitItem)