from django.urls import path

from main import views

app_name = 'main'

urlpatterns = [
    path('', views.index, name='index'),
    path('feedback_shift_generator/', views.feedback_shift_generator, name='feedback_shift_generator'),
    path('create_feedback_shift_generator/', views.create_feedback_shift_generator, name='create_fsg'),

    path('matrix_shift_register/', views.matrix_shift_register, name='matrix_shift_register'),
    path('create_matrix_shift_register/', views.create_matrix_shift_register, name='create_msg'),
]
