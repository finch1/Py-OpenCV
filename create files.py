
path = "Python OpenCV\\"
name = "ex"
ext = ".py"
preamble = "# Date: "

for i in range(10, 21):
    fullname = name + str(i) + ext
    # files and file writing
    # open a file
    myFile = open(path + fullname, "w")
    # erase whatever is in the file and write new
    myFile.write(preamble)
    myFile.close()


