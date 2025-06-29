
from django.db import models
from django.contrib.auth.models import User
from django.db import models
import uuid
# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='profile_images/', default='profile_images/default.jpg')
    
    def __str__(self):
        return f'{self.user.username} Profile'

# Add these new models for the chatbot functionality
class ChatSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Chat {self.session_id[:8]}... ({self.created_at.strftime('%Y-%m-%d')})"

class ChatMessage(models.Model):
    chat_session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    is_user = models.BooleanField(default=True)  # True for user messages, False for bot
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        sender = "User" if self.is_user else "Bot"
        return f"{sender}: {self.content[:30]}..."