css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #433C32;
    box-shadow: 0 0 0.5rem #433C32;
}
.chat-message.bot {
    background-color: #867964;
    box-shadow: 0 0 0.5rem #867964;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow: 0 0 1rem #867964;
  border: 5px solid #CE4125;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/15/d2/64/15d264bb4fdcc4bd07077053791e920c.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/ce/1e/35/ce1e35fc86964a51b6fe32e0856aa51d.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
