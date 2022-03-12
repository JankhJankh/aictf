import pickle
with open("./picklemodel", "rb") as file:
  pickle_data = file.read()
  model = pickle.loads(pickle_data)

model.save('./model.h5')