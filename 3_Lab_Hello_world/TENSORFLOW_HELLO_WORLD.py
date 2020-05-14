#TENSORFLOW'S HELLO WORLD
#TENSORFLOW'S HELLO WORLD------------------------
import tensorflow as tf

#Building a Graph----------------------------------------------------------
#As we said before, TensorFlow works as a graph computational model. Let's create our first graph which we named as graph1.
graph1 = tf.Graph()

'''
Now we call the TensorFlow functions that construct new tf.Operation and tf.Tensor objects and add them to the graph1. 
As mentioned, each tf.Operation is a node and each tf.Tensor is an edge in the graph.

Lets add 2 constants to our graph. For example, calling tf.constant([2], name = 'constant_a') adds a single tf.Operation to the default graph. 
This operation produces the value 2, and returns a tf.Tensor that represents the value of the constant.

Notice: tf.constant([2], name="constant_a") creates a new tf.Operation named "constant_a" and returns a tf.Tensor named "constant_a:0".
'''
with graph1.as_default():
    a = tf.constant([2], name = 'constant_a')
    b = tf.constant([3], name = 'constant_b')

#Lets look at the tensor a.
a

#As you can see, it just show the name, shape and type of the tensor in the graph. We will see it's value when we run it in a TensorFlow session.
# Printing the value of a
sess = tf.Session(graph = graph1)
result = sess.run(a)
print(result)
sess.close()

#After that, let's make an operation over these tensors. The function tf.add() adds two tensors (you could also use c = a + b). 
with graph1.as_default():
    c = tf.add(a, b)
    #c = a + b is also a way to define the sum of the terms

#Then TensorFlow needs to initialize a session to run our code. Sessions are, in a way, a context for creating a graph inside TensorFlow. 
#Let's define our session:
sess = tf.Session(graph = graph1)

#Let's run the session to get the result from the previous defined 'c' operation:
result = sess.run(c)
print(result)

#Close the session to release resources:
sess.close()

#To avoid having to close sessions every time, we can define them in a with block, so after running the with block the session will close automatically:
with tf.Session(graph = graph1) as sess:
    result = sess.run(c)
    print(result)
#Even this silly example of adding 2 constants to reach a simple result defines the basis of TensorFlow. 
#Define your operations (In this case our constants and tf.add), and start a session to build a graph.

#Defining multidimensional arrays using TensorFlow---------------------------------------------------------------------------
#Now we will try to define such arrays using TensorFlow:
graph2 = tf.Graph()
with graph2.as_default():
    Scalar = tf.constant(2)
    Vector = tf.constant([5,6,2])
    Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
with tf.Session(graph = graph2) as sess:
    result = sess.run(Scalar)
    print ("Scalar (1 entry):\n %s \n" % result)
    result = sess.run(Vector)
    print ("Vector (3 entries) :\n %s \n" % result)
    result = sess.run(Matrix)
    print ("Matrix (3x3 entries):\n %s \n" % result)
    result = sess.run(Tensor)
    print ("Tensor (3x3x3 entries) :\n %s \n" % result)

#tf.shape returns the shape of our data structure.
Scalar.shape

Tensor.shape

#Now that you understand these data structures, I encourage you to play with them using some previous functions to see how they will behave, 
#according to their structure types:
graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph =graph3) as sess:
    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print ("Defined using normal expressions :")
    print(result)

'''
With the regular symbol definition and also the TensorFlow function we were able to get an element-wise multiplication, also known as Hadamard product.

But what if we want the regular matrix product?

We then need to use another TensorFlow function called tf.matmul():
'''
graph4 = tf.Graph()
with graph4.as_default():
    Matrix_one = tf.constant([[2,3],[3,4]])
    Matrix_two = tf.constant([[2,3],[3,4]])

    mul_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session(graph = graph4) as sess:
    result = sess.run(mul_operation)
    print ("Defined using tensorflow function :")
    print(result)

#Variables------------------------------------------------------------------------------------------------
v = tf.Variable(0)

'''
Let's first create a simple counter, a variable that increases one unit at a time:

To do this we use the tf.assign(reference_variable, value_to_update) command. tf.assign takes in two arguments, the reference_variable to update, 
and assign it to the value_to_update it by.
'''
update = tf.assign(v, v+1)

#Variables must be initialized by running an initialization operation after having launched the graph. 
#We first have to add the initialization operation to the graph:
init_op = tf.global_variables_initializer()

'''
We then start a session to run the graph, first initialize the variables, then print the initial value of the state variable, 
and then run the operation of updating the state variable and printing the result after each update:
'''
with tf.Session() as session:
    session.run(init_op)
    print(session.run(v))
    for _ in range(3):
        session.run(update)
        print(session.run(v))

#Placeholders----------------------------------------------------------------------------------------------------
#So we create a placeholder:
a = tf.placeholder(tf.float32)

#And define a simple multiplication operation:
b = a * 2

'''
Now we need to define and run the session, but since we created a "hole" in the model to pass the data, 
when we initialize the session we are obligated to pass an argument with the data, otherwise we would get an error.

To pass the data into the model we call the session with an extra argument feed_dict in which 
we should pass a dictionary with each placeholder name followed by its respective data, just like this:
'''
with tf.Session() as sess:
    result = sess.run(b,feed_dict={a:3.5})
    print (result)

#Since data in TensorFlow is passed in form of multidimensional arrays we can pass any kind of tensor through the placeholders 
#to get the answer to the simple multiplication operation:
dictionary={a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }

with tf.Session() as sess:
    result = sess.run(b,feed_dict=dictionary)
    print (result)

#Operations --------------------------------------------------------------------------------------------------
'''
Operations are nodes that represent the mathematical operations over the tensors on a graph. 
These operations can be any kind of functions, like add and subtract tensor or maybe an activation function.

tf.constant, tf.matmul, tf.add, tf.nn.sigmoid are some of the operations in TensorFlow. 
These are like functions in python but operate directly over tensors and each one does a specific thing. 
'''
graph5 = tf.Graph()
with graph5.as_default():
    a = tf.constant([5])
    b = tf.constant([2])
    c = tf.add(a,b)
    d = tf.subtract(a,b)

with tf.Session(graph = graph5) as sess:
    result = sess.run(c)
    print ('c =: %s' % result)
    result = sess.run(d)
    print ('d =: %s' % result)

