class dictionary:
    def __init__(self):
        self.size = 10
        self.map = [None] * self.size #array to store the data in
        self.counter =0
    
    # to pick where to store the key
    def get_hash(self,key):
        hash=0
        for char in str(key):
            hash+=ord(char) #no or ascii code of the car
        return hash % self.size
    


    def add(self,key,value):
        self.counter=self.counter+1
        key_hash = self.get_hash(key)
        key_value = [key,value]

        #If this place in map in empty then create a list in this place, first place will be a list of the given key and value
        if self.map[key_hash] is None:
            self.map[key_hash] =list ([key_value])
            return True
        else:
            
            for kv in self.map[key_hash]:
                #if he wants to update the value of a key
                if kv[0] == key:
                    kv[1]=value
                    return True
                #if he doesnt want to updat the value then append the new key to the list
                self.map[key_hash].append(key_value)
                return True

    def get(self , key):
        key_hash = self.get_hash(key)

        #check if this index generated search by it in the map and see if  its content is not empty
        if self.map[key_hash] is not None:
            #start searching in that place of index in the placed pairs which of them has my key
            for kv in self.map[key_hash]:
                if kv[0] == key:
                    return kv[1]
            return None #because the place may have many pair but my key doesnt exist

    def delete (self, key):
        self.counter=self.counter-1
        key_hash = self.get_hash(key)
        if self.map[key_hash] is not None:
            #cannt be for loop with kv as get bec kv is a list not int or index
            i=0
            for kv in  self.map[key_hash]:
                if kv[0] == key:
                    self.map[key_hash].pop(i)
                    return True
                i=i+1
            return False
    
    def print (self):
        for i in self.map:
            if i is not None:
                print(i)


# mydict = dictionary()
# if(mydict.counter == 0):
#     print("zero")
# mydict.add(3,[2.5,2])
# if(mydict.counter == 0):
#     print("zero")
# else:
#     print("one")
#mydict.add('T','Tasnim')
#mydict.add ('s', 333)
#x=mydict.delete('T')
#y= mydict.get('s')
# mydict.print()
#print(x)
#print(y)


