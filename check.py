import os

age_prototxt = os.path.join(os.getcwd(), "model", "deploy_age.prototxt")
age_caffemodel = os.path.join(os.getcwd(), "model", "age_net.caffemodel")

print("Prototxt exists:", os.path.isfile(age_prototxt))
print("Caffemodel exists:", os.path.isfile(age_caffemodel))
