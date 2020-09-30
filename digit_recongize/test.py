class Animals:
    def welcome(self):
        print("Hello  am Tom")

class dog(Animals):
    def welcome(self):
        super().welcome()
        print("I am Dog")

for i in range(4):
    print(i)

d = dog()
d.welcome()
