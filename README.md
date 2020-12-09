# project-1
# Python Data Types

Data Types vary in characteristics, features and restrictions and therefor determine their usefulness in certain cases. They can be categoriesed as ...

### Boolean
The most basic data type in python is probably 'bool'. It stands for a boolean value of either 'True' or 'False'.
The output from a test which checks, whether an expression ist true or false can be 0 or 1, though.

### Numeric
Numeric data types are integer (int), floating point (float) or complex. The type 'complex' is less interesting in our context here and will not be further discussed.
Integers are whole numbers, either positive (+13) or negative (-67). 
'float' can also hold positive and negative values. But they do not need to be whole numbers and always contain a decimal point. 7 : 2 would produce the float value of 3.5.
The types int and float are interchangeable. And int can be turned into a float and vice versa. When changing a value of float into an int, the decimal part will be omitted: 54.7 (float) will become 54 (int). 

### Sequences
Whenever ordering of elements is crucial, sequences are very useful. Their elements are indexed and can be accessed in the order they were created. Indexing starts with 0 (zero). The first elemment in a sequences would be [0]. Negative indexing is also possible. [-2] would address the second last element in a sequence.

    * strings (str)
      Strings are immutable. Their content cannot be changed once created. Strings hold unicade characters 
      and are declared using either single (') or double quotation marks ("). The value of 1 inside these 
      would represent a string and not an integer. 
      However it would still be recongnizable as a digit using the isdigit() function.
          isdigit('7654') returns 'True'. Arithmetic operations are also possible with strings.

    * lists (list)
      Lists are declared using square brackets []. Lists are mutable, i.e. their items can be changed after 
      creation. Unlike strings.
      They are heterogenous and can contain any type data as items, like strings or integers, for example. 
      Items can also be duplicates. aaa = [ 1, 65.786, 'Me.' ] is valid.

    * tuples (tuple)
      Tuples can be regarded as immutable lists. They can also contain data of different types and duplicates 
      but are not changeable after creation. And this makes them a useful tool to create keys for dictionaries, 
      which we will discuss later on. 
      The use of parentheses will declare a tuple: bbb = ( 1, 65.786, 'Me.' ) 
      Indeed, any assignment of values seperated by commas will create a tuple in python.
      Tuples can be sorted like bbb.sorted() . However, since they are immutable, 
      this would create a sorted copy of the original.

There are more types of sequences in python which are not interesting for us as of now.


### Containers
Unlike sequences, containers are not indexed and therefor do not know about any ordering of their elements.

    * sets (set)
      Elements in a set may be of various data types but must be unique. One would use curly brackets to declare 
      a set: ccc = { "Python", 4.789, ('a', 23)} would be fine. ccc = { "Python", "Python", 4.789, ('a', 23)} 
      would not produce an error but would only allow one occurrance of the string "Python" be an item in this set.
      A set is mutable, items can be deleted or added. But not into something that is again mutable 
      (like a list or another set).
      The function sorted() is also available for sets. Be aware that sorted(ccc) does not change anything in the 
      set ccc or create a copy of it but returns a list with the sorted elements.
      Methods like intersection() and difference()  and more can be used with sets to determine, 
      whether elements are contained in different sets.

    * dictionaries (dict)
      Dictionaries are special since the do hold single values as items like the other types but pairs as key : value. 
      As mentioned above, tuples can be used to genarate keys. Curly brackets are also used here for declaration. 
      Indeed, ddd = {} creates an empty dictionary, not an empty set.
      But dictinaries hold key : value pairs and can therefor not be confused with sets. eee = { 'Name':42, 1.9:'no' }
      We cannot use an index here to access a certain value (like we would in a list) but we can use its corresponding key. 


# Decision Making
In a program or script sometimes we want to execute code only when certain conditions are met. 
### If-Statement
if is followed by an expression. Relational operators are used to build up such an expression. (==, > and others).
The line is terminated by a colon (:).
Whenever an expression is True, the code beginning in the following line is executed. This line has to be indented.
Usually by 4 spaces. Should more code be executed, it would follow on the next lines which then indented by the same degree.
Nothing will be executed should the expression be False.
```
if a == 5:
    print("Value of Variable a is 5.")
```
#### If-else-Statement
But we can force one of two commands to be executed depending on the truthiness of the expression.
Indentation determines the code block.
```
if a == 5:
    print("Value of variable a is 5.")
else:
    print("Value of variable a is not 5.")
```

#### Elif
It is possible to check more than one condition using elif. It can be used several times between if and else.
```
if a == 5:
    print("Value of variable a is 5.")
elif a == 3:
    print("Value of variable a is 3.")
elif a == 2:
    print("Value of variable a is 2.")
else:
    print("Value of variable a is neither 5, 3 nor 2.")
```

#### Nesting
If-statements can also be combined. You can let another test follow should an expression be True.
```
if a < 7:
    if a > 5:
        print("a is 6.")
    print("a is less than 7.")
else:
    print("a is greater than 7.")
```
# Loops
To automate the execution of several statements loops are used. There are two different types.
### For-Loop
To iterate over a sequence we would use a for loop.
```
for i in {2,3,3,4}:
    print(i)
```
