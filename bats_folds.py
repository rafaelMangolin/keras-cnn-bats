import os
import shutil

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def getClass(number):
	number = int(number)
	if number<=1000:
		return 'one'
	elif number>1000 and number<=2000:
		return 'two'
	elif number>2000 and number<=3000:
		return 'three'
	elif number>3000:
		return 'four'


def copyFiles(foldName,mode,array):
	i=0
	for e in array:
 		shutil.copy2('./IMG/{0}.jpg'.format(e), 
 			'./{0}/{1}/{2}/{2}.{3}.jpg'.format(foldName,mode,getClass(e),i))
 		i=i+1

def createFolders(foldName):
	createFolder(foldName)
	createFolder("./{0}/Test".format(foldName))
	createFolder("./{0}/Test/one".format(foldName))
	createFolder("./{0}/Test/two".format(foldName))
	createFolder("./{0}/Test/three".format(foldName))
	createFolder("./{0}/Test/four".format(foldName))
	createFolder("./{0}/Train".format(foldName))
	createFolder("./{0}/Train/one".format(foldName))
	createFolder("./{0}/Train/two".format(foldName))
	createFolder("./{0}/Train/three".format(foldName))
	createFolder("./{0}/Train/four".format(foldName))

def createFold(foldNum,foldName):
	createFolders(foldName)
	file_object = open("division{0}.txt".format(foldNum), "r")
	content = file_object.read()
	array = content.replace("\n","").split(",")
	train = array[:3200]
	test = array[3200:4000]
	copyFiles(foldName,'Train',train)
	copyFiles(foldName,'Test',test)




createFold(1,"one")
createFold(2,"two")
createFold(3,"three")
createFold(4,"four")
createFold(5,"five")