
class AnnotationInfo:
    def __init__(self):
        self.classInfo = {'1': 'tree', '2': 'person', '3': 'house'}
        self.typeOfChildren = 'class_5_1_1'
    def setTypeOfChildren(self, type):
        self.typeOfChildren = type
    def getClassName(self, classNumber):
        name = []
        for index in classNumber:
            values = self.classInfo.values()
            values_list = list(values)
            name.append(values_list[index])
        return name
    def getInfo(self, classNumber):
        if self.typeOfChildren == 'class_5_1_1':
            self.classInfo = {'1': 'tree', '2': 'person', '3': 'house'}
            className = self.getClassName(classNumber)
            return className
        elif self.typeOfChildren == 'class_5_1_2':
            self.classInfo = {'1': 'tree', '2': 'person', '3': 'house'}
            className = self.getClassName(classNumber)
            return className
        elif self.typeOfChildren == 'class_6_1_1':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            className = self.getClassName(classNumber)
            return className
        elif self.typeOfChildren == 'class_6_1_2':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            className = self.getClassName(classNumber)
            return className
        elif self.typeOfChildren == 'class_7_1_1':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            className = self.getClassName(classNumber)
            return className
        elif self.typeOfChildren == 'class_7_1_2':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            className = self.getClassName(classNumber)
            return className

if __name__ == "__main__":
    a = AnnotationInfo()
    print(a.getInfo([1,2,0]))

