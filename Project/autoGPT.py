import bardapi
import os
import re

# set your __Secure-1PSID value to key
token = 'WwiPoDPj7P_yu6yztDc21z_13IzJjpGSpv4p_6c81CZ792wBDL9GF1TfmC0vWRNPZh6MDA.'

# set your input text

input_text = "are pushups good for you"
name = "Moaksh Kakkar"
age= 18
height_cms= 195
diet_plan_bukin = f"my name is {name} my age is {age} my height is {height_cms} in cms now make me a diet plan for bulking"


input_text = "how to fix my sleep schedule"
working_hours =  8
sleep_schedule = f"my name is {name} my age is {age} my working hours  is {working_hours}  now make me a sleep schedule"


input_text = "how to get rid of injuries"
age = 23
workout_time_hours = 2
avoid_injuries = f" my age is {age} my workout time is {workout_time_hours}  now say me how to avoid injuries"

input_text = "what is the beginner weight to start with "
age = "23"
body_weight = 56
weight_to_start = f" my age is {age} my body_weight is {body_weight} now suggest me the weight to start with"

input_text = " how many reputation for each exersice to increase muscle mass"
age = "34"
body_weight = 65
height = "173"
reputations = f"person with age{age} and a height {height} with a body weight {body_weight} now suggest me how many repetation to increase the muscle mass"

input_text = "how many days a week should i workout "
age = "33"
health_condition = "diabetes"
days_of_work = f"person with age {age} and with a health condition of {health_condition} now suggest me how  many days a week should i workout"

input_text =  " morning or eveming which is the better time to workout "
job_hours = "8"
perfect_time = f"a person with job_hours{job_hours} suggest me the better time to workout morninhg or evening "

input_text = "which are the healthy fats to have for good performance "
age = " 32"
gender = " male"
health_conditions = "heart diasease"
good_fat = f" a person with health condition {health_conditions} age {age} and gender{gender} suggest me the best healthy fats to eat "
# Send an API request and get a response.
response = bardapi.core.Bard(token).get_answer(diet_plan_bukin)
# print(response)
lines = response['content'].split('\n')
for i in range(len(lines)):
   print(lines[i])

