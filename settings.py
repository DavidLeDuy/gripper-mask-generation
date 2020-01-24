import json



_name = ""
_classes = []
settings_dict = {}

try:
    f = open("settings.json")
    chose = input("File already exists. Do you want to overwrite it? y/n ")
    if chose == "y" or chose == "Y":
        pass
    else:
        exit()
except FileNotFoundError:
    pass
finally:
    f.close()


print("This is the setup to train your own model")
print("-----------------------------------------------------------")


_name = input("Enter name of your Project: ")

print("-----------------------------------------------------------")

classesInput = input("Enter classes of your Project separated by commas: ")
_classes = classesInput.replace(" ","").split(",")

print("-----------------------------------------------------------")
_c_nr = 0
for klasse in _classes:
    _c_nr +=1

settings_dict = {
    "name": _name,
    "classes": _classes,
    "c_nr": _c_nr
}

with open("settings.json","w") as file:
    json.dump(settings_dict,file)


print("This is your project name: ", _name)
print("These are your classes: ", _classes)

