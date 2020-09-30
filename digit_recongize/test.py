class Animals():
    def welcome(self):
        print("Hello  am Tom")

class dog(Animals):
    def welcome(self):
        super.welcome()


a = dog()
a.dog()
