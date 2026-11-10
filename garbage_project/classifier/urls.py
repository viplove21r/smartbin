from django.urls import path
from . import views

urlpatterns = [
    path("", views.predict_waste, name="predict_waste"),
    path("dataset/", views.dataset_view, name="dataset_view"),
    path("model-info/", views.model_info, name="model_info"),

    path("compost/", views.compost_bin, name="compost_bin"),

    path(
        "compost/add/",
        views.add_compost_item,
        name="add_compost_item"
    ),

    path(
        "compost/create-pit/",
        views.create_compost_pit,
        name="create_compost_pit"
    ),

    path(
        "compost/remove/<int:item_id>/",
        views.remove_compost_item,
        name="remove_compost_item"
    ),
    path(
    "compost/delete-pit/<int:pit_id>/",
    views.delete_compost_pit,
    name="delete_compost_pit"
),
path("submit-feedback/", views.submit_feedback, name="submit_feedback"),
path("my-waste/", views.my_waste, name="my_waste"),
]