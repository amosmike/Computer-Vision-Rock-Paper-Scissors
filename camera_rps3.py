# Repeat effort
import time
from turtle import color
import cv2
from keras.models import load_model
import numpy as np
import random
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Import manual_rps functions
from manual_rps import comp_wins, user_wins, get_computer_choice, get_user_choice, get_winner

# Globals
options = ['rock', 'paper', 'scissors', 'nothing']

    # Text
font = cv2.FONT_HERSHEY_SIMPLEX
color = [78,13,147]
thickness = 2
cvline = cv2.LINE_AA

    # Time
t0 = time.time()

# computer_wins = []
user_list = []

comp_pick = ["rock", "paper", "scissors"]
comp = random.choice(comp_pick)

# Make game last 5 seconds
# timeout = time.time() + 5   # 5 seconds from now
player = input("Please enter your name: ")

start_time = time.time()
seconds = 1


user_score = 0
comp_score = 0
score = [user_score, comp_score]

def play():

    play_again = "y"
    
    while play_again == "y":
        current_time = time.time()
        elapsed_time = current_time - start_time
        countdown = seconds - elapsed_time

        if elapsed_time > seconds:
            #print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
            break

        print("Time remaining = " + str("{:.2f}".format(countdown)) + " seconds")

        # time.time() < timeout
        
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
        # Press q to close the window

        # print(prediction)
        # getting largest number prediction
        # print(options[np.argmax(prediction)])

        # Display the game status 
        cv2.putText(frame, player, (250, 250), font, 1, color, thickness, cvline)
        cv2.putText (frame, f"Computer", (480, 30), font, 1, color, thickness, cvline)
        # cv2.putText (frame, f"I{score[0]}| {(score[1]}]", (280, 30), font, 1, color, thickness, cvline)
        # cv2.putText (frame, f"Round {game_round}", (10, 460), font, 1, color, thickness, cvline)
        cv2.imshow('frame', frame)

        # Get time
        #t1 = (time.time())
        #print("Time elapsed = " + str("{:.2f}".format(elapsed_time)) + " seconds")
        
        user_list.append(prediction)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

    out_array = [sum(x) for x in zip(*user_list)]
    print(out_array)
    #out_arr = np.add(user_choice)
    #print(user_choice)

    user_choice = options[np.argmax(prediction)]

    print("Player: " + user_choice)
    print("Computer: " + comp)

    get_winner(comp, user_choice, score)

    while True:
        play_again = input("Would you like another game? (y/n) \n")
        if play_again == 'y':
            return play()
        elif play_again == 'n':
            break
        else:
            print("??HUH??")
            return True



play()



# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()

