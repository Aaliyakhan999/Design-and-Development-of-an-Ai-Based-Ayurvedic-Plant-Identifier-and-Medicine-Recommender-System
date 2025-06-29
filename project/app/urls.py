
from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('', views.home, name='home'),
   path('chat/', views.chat_view, name='chat_view'), 
   path('recipe/', views.recipe_view, name='recipe_view'), 

    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup, name='signup'),
    path('profile/', views.profile, name='profile'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('prediction/', views.prediction, name='prediction'),
    path('team/', views.team, name='team'),
    path('upload_profile_image/', views.upload_profile_image, name='upload_profile_image'),
]
