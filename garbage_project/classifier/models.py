from django.db import models
from django.utils import timezone
from datetime import timedelta

from django.conf import settings


class CompostItem(models.Model):
    WASTE_CHOICES = [
        ("paper", "Paper"),
        ("cardboard", "Cardboard"),
    ]

    waste_type = models.CharField(
        max_length=20,
        choices=WASTE_CHOICES
    )

    added_at = models.DateTimeField(auto_now_add=True)

    is_in_pit = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.waste_type} - {self.id}"


class CompostPit(models.Model):
    name = models.CharField(max_length=100)

    created_at = models.DateTimeField(auto_now_add=True)

    start_time = models.DateTimeField()

    end_time = models.DateTimeField()

    status = models.CharField(
        max_length=20,
        default="Active"
    )

    def save(self, *args, **kwargs):
        if not self.start_time:
            self.start_time = timezone.now()

        if not self.end_time:
            self.end_time = self.start_time + timedelta(days=30)

        super().save(*args, **kwargs)

    @property
    def is_completed(self):
        return timezone.now() >= self.end_time

    @property
    def remaining_seconds(self):
        remaining = self.end_time - timezone.now()
        return max(0, int(remaining.total_seconds()))

    @property
    def elapsed_seconds(self):
        elapsed = timezone.now() - self.start_time
        return max(0, int(elapsed.total_seconds()))

    def __str__(self):
        return self.name


class CompostPitItem(models.Model):
    pit = models.ForeignKey(
        CompostPit,
        on_delete=models.CASCADE,
        related_name="items"
    )

    item = models.ForeignKey(
        CompostItem,
        on_delete=models.CASCADE,
        related_name="pit_assignments"
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["item"],
                name="one_pit_per_compost_item"
            )
        ]




class WasteScan(models.Model):

    WASTE_CHOICES = [
        ("cardboard", "Cardboard"),
        ("glass", "Glass"),
        ("metal", "Metal"),
        ("paper", "Paper"),
        ("plastic", "Plastic"),
        ("trash", "Trash"),
    ]

    FEEDBACK_CHOICES = [
        ("correct", "Correct"),
        ("incorrect", "Incorrect"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )

    image = models.ImageField(
        upload_to="waste_scans/",
        null=True,
        blank=True
    )

    predicted_class = models.CharField(
        max_length=20,
        choices=WASTE_CHOICES
    )

    confidence = models.FloatField()

    feedback = models.CharField(
        max_length=10,
        choices=FEEDBACK_CHOICES,
        null=True,
        blank=True
    )

    actual_class = models.CharField(
        max_length=20,
        choices=WASTE_CHOICES,
        null=True,
        blank=True
    )

    embedding = models.JSONField(
    null=True,
    blank=True
)

    scanned_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_class} - {self.scanned_at}"