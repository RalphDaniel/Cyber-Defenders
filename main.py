a

Explanation:

In
the
above
snippet
of
code, we
have
specified
the
path
to
the
image
files
of
the
License
number
plate
using
the
OS
module.We
have
also
defined
two
empty
lists as NP_list and predicted_NP.We
have
then
appended
the
actual
number
plate
to
the
list
using
the
append()
function.We
then
used
the
OpenCV
module
to
read
each
number
plate
image
file and stored
them in the
NP_img
variable.We
have
then
passed
each
number
plate
image
file
to
the
Tesseract
OCR
engine
with the help of the Python library wrapper.We have then got back predicted_res for the number plate and append it in a list and compare it with the genuine one.

Now, since
we
have
the
plates
predicted
but
we
don
"t know the prediction. So, in order to view the data and prediction, we will perform a bit of visualization, as shown below. We will also be estimating the accuracy of the prediction without the help of any core function.

print("Original Number Plate", "\t", "Predicted Number Plate", "\t", "Accuracy")
print("--------------------", "\t", "-----------------------", "\t", "--------")


def estimate_predicted_accuracy(ori_list, pre_list):
    for ori_plate, pre_plate in zip(ori_list, pre_list):
        acc = "0 %"
        number_matches = 0
        if ori_plate == pre_plate:
            acc = "100 %"
        else:
            if len(ori_plate) == len(pre_plate):
                for o, p in zip(ori_plate, pre_plate):
                    if o == p:
                        number_matches += 1
                acc = str(round((number_matches / len(ori_plate)), 2) * 100)
                acc += "%"
        print(ori_plate, "\t", pre_plate, "\t", acc)


estimate_predicted_accuracy(NP_list, predicted_NP)

print("Original Number Plate", "\t", "Predicted Number Plate", "\t", "Accuracy")
print("--------------------", "\t", "-----------------------", "\t", "--------")


def estimate_predicted_accuracy(ori_list, pre_list):
    for ori_plate, pre_plate in zip(ori_list, pre_list):
        acc = "0 %"
        number_matches = 0
        if ori_plate == pre_plate:
            acc = "100 %"
        else:
            if len(ori_plate) == len(pre_plate):
                for o, p in zip(ori_plate, pre_plate):
                    if o == p:
                        number_matches += 1
                acc = str(round((number_matches / len(ori_plate)), 2) * 100)
                acc += "%"
        print(ori_plate, "\t", pre_plate, "\t", acc)


estimate_predicted_accuracy(NP_list, predicted_NP)

Explanation:

In
the
above
snippet
of
code, we
have
defined
a
function
calculating
the
predicted
accuracy.Within
the
function, we
used
the
for -loop to iterate through the list of original number plates and the predicted ones and checked if they matched.We have also checked the accuracy on the basis of the number's length for getting better and appropriate results.

We
can
observe
that
the
Tesseract
OCR
engine
mostly
predicts
all
of
the
license
plates
correctly
with a rate of 100 % accuracy.The Tesseract OCR engine predicted incorrectly for the number plates, and we will apply the image processing technique on those number plate files and pass them to the Tesseract OCR again.We can increase the accuracy rate of the Tesseract Engine for the number plates of the incorrectly predicted number plates by applying the techniques of image processing.

*Let
us
consider
the
following
snippet
of
code
to
understand
the
technique
of
Image
Processing.

import matplotlib.image as mpimg

for img in os.listdir("D://Python//License_Plate"):
    test_NP = mpimg.imread("W5KHN.jpg")

plt.imshow(test_NP)
plt.axis('off')
plt.title('W5KHN license plate')
plt.show()

Out
put

Explanation:

In
the
above
snippet
of
code, we
have
imported
the
image
module
from the matplotlib

library and used
the
for -loop to extract the image from the designated folder.We have then used the imread() function to read the extracted image.We have then used the plot module of the matplotlib library to display the image for the users.

Image
Resizing: We
can
resize
the
image
file
by
a
factor
of
2
x in both
the
horizontal and vertical
directions
with the help of resize.
Converting
to
Gray - scale: Then, we
can
convert
the
resized
image
file
to
grayscale in order
to
optimize
the
detection and reduce
the
number
of
colours
available in the
image
drastically, which
will
allow
us
to
detect
the
number
plates
easily.
Denoising
the
Image: We
can
use
the
Gaussian
Blur
technique
to
denoise
the
images.It
makes
the
edges
of
the
image
clearer and smoother, making
the
characters
more
readable.
Let
us
consider
the
following
example
to
understand
the
same.

# image resizing
resize_test_NP = cv2.resize(
    test_NP, None, fx=2, fy=2,
    interpolation=cv2.INTER_CUBIC)

# converting image to grayscale
grayscale_resize_test_NP = cv2.cvtColor(
    resize_test_NP, cv2.COLOR_BGR2GRAY)

# denoising the image
gaussian_blur_NP = cv2.GaussianBlur(
    grayscale_resize_test_NP, (5, 5), 0)

Explanation:

In
the
above
snippet
of
code, we
have
some
tools
of
the
OpenCV
module
to
resize
the
image, convert
it
into
grayscale, and denoise
the
image.

Once
the
above
steps
are
complete, we
can
pass
the
transformed
license
plate
file
to
the
Tesseract
OCR
engine and view
the
predicted
result.

The
same
can
be
observed in the
following
snippet
of
code.

new_pre_res_W5KHN = pytesseract.image_to_string(gaussian_blur_NP, lang='eng')
filter_new_pre_res_W5KHN = "".join(new_pre_res_W5KHN.split()).replace(":", "").replace("-", "")
print(filter_new_pre_res_W5KHN)

--------------lahat
yan
i
save
project as anypy.py

eto
susunod
iba
nato

--------------lahat
naman
neto
na
susunod
na
codes
isave as func.py

# importing the required modules
import os
import json
import sys
from datetime import datetime


# defining an inference function
def infer(person_name="", tag=True):
    '''
    if the present year is 2021, then inference function will execute properly, else it fails.
    Here the attribute variable contains the string version of the date in MM-DD-YYYY format
    '''

    print("Hello " + person_name + ", the inference function has been initiated successfully")

    atr = str(datetime.now().strftime('%m-%d-%Y'))
    resp = "Your license has been expired, please contact us."
    expiration_year = int(2023)

    try:
        assert int(atr.split('-')[-1]) == expiration_year, resp
    except AssertionError as e:
        print(resp)
        sys.exit()

        # if the above assertion is True, it will reach until this point,
    # otherwise it will stop in the previous line.

    if tag:
        print("Inference function has been done properly!")
        return True
    else:
        return False


if _name_ == "_main_":
    _ = infer(person_name="Peter Parker")

    '''  
    Function outputs,  
    Case 1: if expiration_year = int(2021)  
    Hello Peter Parker, the inference function has been intiated successfully  
    Inference function has been done properly!  
    [Finished in 0.2s]  

    Case 2: if expiration_year = int(2022)  
    Hello Peter Parker, the inference function has been intiated successfully  
    Inference function has been done properly!  
    [Finished in 0.2s]  

    Case 3: if expiration_year = int(2023)  
    Hello Peter Parker, the inference function has been intiated successfully  
    You license has been expired, please contact us.  
    [Finished in 0.2s]  
    '''
-----ang
output
nyan
message
lng

Explanation:

In
the
above
snippet
of
code, we
have
imported
some
required
modules.We
then
defined
an
inference
function as infer(), which
accepts
two
parameters - person_name and tag = True.We
have
then
printed
a
statement
for the user initiating the inference function.Later, we defined a variable as atr that stores the current date and a string variable as resp.We have also assigned another variable as expiration_year to 2023. We have then used the try-exception method in order to handle any exception if raised.At last, we have used the if - else conditional statements to print a statement as per the situation.At last, we have assigned _name_ to "_main_" to execute the inference function.

Now, let
us
save
this
python
file in a
folder.

Step
1: Installing
the
pyarmor
package

We
can
install
the
pyarmor
package
using
the
pip
installer as shown
below:

Syntax:

$ pip
install
pyarmor

sa
cmd
lng
to

Step
2: Encrypting
the
python
file

We
can
encrypt
the
file
by
typing
the
following
command in the
command
prompt.

Syntax:

$ pyarmor
obfuscate - -restrict = 0 < filename >
Now, let
us
implement
the
above
command
on
the
func.py
file.

$ pyarmor
obfuscate - -restrict = 0
func.py

Now,
if we open the folder consisting of the original func.py file, we will observe a new subfolder created known as dist.

Within
the
dist
folder, we
will
find
another
folder as pytransform and an
encrypted
func.py
file.

Now, let
us
see
the
contents
inside
this
file.

File: func.py(encrypted)

from pytransform import pyarmor_runtime

pyarmor_runtime()
_pyarmor__(__name_, _file_, b'\x50\x59\x41\x52\x4d\x4f\x52\x00\...', 2)

Importing
the
Inference
function
Once
we
are
done, until
this
section, now
let
us
try importing this encrypted func.py in a new python file known as new.py, which we created within the dist folder.

The
mandatory
keys
that
allow
us
to
decrypt
the
func.py
at
run - time
are
taken
care
of
using
pyarmor.Its
presence
exists in the
pytransform
folder;
hence, creating
the
code
unreadable
to
other
's eyes.

However,
if we would like to do some modifications to the actual func.py script, we have to start from step 1 continue following the same steps.

---------------bago
na
ulit
to
save as new.py
per
same
folder
lng
sa
python

# importing the inference function definition inside the func.py file
from func import infer

_ = infer(person_name="Tony Stark")

Output:

Hello
Tony
Stark, the
inference
function
has
been
initiated
successfully
Your
license
has
been
expired, please
contact
us.
Explanation:

In
the
above
snippet
of
code, we
have
imported
the
inference
function
of
the
func.py in the
new
python
file
that
we
created as new.py.We
have
then
executed
that
function
with the same configuration as of the func.py.

Now
let
's list out the methods that can convert a string to a dictionary.

Using
loads()
Using
literal_eval()
Using
Generator
Expressions

# using json()
import json

# initialising the string
string_1 = '{"subj1":"Computer Science","subj2":"Physics","subj3":"Chemistry","subj4":"Mathematics"}'
print("String_1 is ", string_1)
# using json.loads()
res_dict = json.loads(string_1)
# printing converted dictionary
print("The resultant dictionary is ", res_dict)

output

String_1 is {"subj1": "Computer Science", "subj2": "Physics", "subj3": "Chemistry", "subj4": "Mathematics"}
The
resultant
dictionary is {'subj1': 'Computer Science', 'subj2': 'Physics', 'subj3': 'Chemistry', 'subj4': 'Mathematics'}
Explanation:

Let
's understand what we have done in the above program-

In
the
first
step, we
have
imported
the
json
module.
After
this, we
have
initialized
the
string
that
we
would
like
to
convert.
Now
we
have
simply
passed
'string_1' as a
parameter in loads().
Finally, in the
last
step, we
have
displayed
the
resultant
dictionary.

Now
we
will
see
how
ast.literal_eval
can
help
us
to
meet
our
objective.

The
following
program
illustrates
the
same -

# convert string to dictionary
# using ast()
import ast

# initialising the string
string_1 = '{"subj1":"Computer Science","subj2":"Physics","subj3":"Chemistry","subj4":"Mathematics"}'
print("String_1 is ", string_1)
# using ast.literal_eval
res_dict = ast.literal_eval(string_1)
# printing converted dictionary
print("The resultant dictionary is ", res_dict)

output

String_1 is {"subj1": "Computer Science", "subj2": "Physics", "subj3": "Chemistry", "subj4": "Mathematics"}
The
resultant
dictionary is {'subj1': 'Computer Science', 'subj2': 'Physics', 'subj3': 'Chemistry', 'subj4': 'Mathematics'}
Explanation:

Let
's understand what we have done in the above program-

In
the
first
step, we
have
imported
the
ast
module.
After
this, we
have
initialized
the
string
that
we
would
like
to
convert.
Now
we
have
simply
passed
'string_1' as a
parameter in literal_eval().
Finally, in the
last
step, we
have
displayed
the
resultant
dictionary.

Let
's study the given program carefully.

# convert string to dictionary
# using generator expressions
# initialising the string
string_1 = "subj1 - 10 , subj2 - 20, subj3 - 25, subj4 - 14"
print("String_1 is ", string_1)
# using strip() and split()
res_dict = dict((a.strip(), int(b.strip()))
                for a, b in (element.split('-')
                             for element in string_1.split(', ')))
# printing converted dictionary
print("The resultant dictionary is: ", res_dict)
print(type(res_dict))

Output:

String_1 is subj1 - 10, subj2 - 20, subj3 - 25, subj4 - 14
The
resultant
dictionary is: {'subj1': 10, 'subj2': 20, 'subj3': 25, 'subj4': 14}
<

class 'dict'>


It
's time to check the explanation of this approach-

In
the
first
step, we
have
declared
a
string
that
has
values
paired
with a hyphen, and each pair is separated with a comma.This information is important since it will act as a great tool in obtaining the desired output.
Further, we
have
used
strip() and split() in the
for loop so that we get the dictionary in the usual format.
Finally, we
have
printed
the
dictionary
we
created and verified
its
type
using
type().

Let
us
have
a
look
at
the
first
approach
of
converting
a
string
to
json in Python.

The
following
program
illustrates
the
same.

# converting string to json
import json

# initialize the json object
i_string = {'C_code': 1, 'C++_code': 26,
            'Java_code': 17, 'Python_code': 28}

# printing initial json
i_string = json.dumps(i_string)
print("The declared dictionary is ", i_string)
print("It's type is ", type(i_string))

# converting string to json
res_dictionary = json.loads(i_string)

# printing the final result
print("The resultant dictionary is ", str(res_dictionary))
print("The type of resultant dictionary is", type(res_dictionary))

Output:

The
declared
dictionary is {'C_code': 1, 'C++_code': 26,
               'Java_code': 17, 'Python_code': 28}
It
's type is <class '
str
'>
The
resultant
dictionary is {'C_code': 1, 'C++_code': 26,
               'Java_code': 17, 'Python_code': 28}
The
type
of
resultant
dictionary is <


class 'dict'>


Explanation:

It
's time to see the explanation so that our logic becomes clear-

Since
here
the
objective is to
convert
a
string
to
json
file
we
will
first
import the

json
module.
The
next
step is to
initialize
the
json
object in which
we
have
the
subject
name as keys and then
their
corresponding
values
are
specified.
After
this, we
have
used
dumps()
to
convert
a
Python
object
to
a
json
string.
Finally, we
will
use
loads()
to
parse
a
JSON
string and convert
it
into
a
dictionary.

Using
eval()
# converting string to json
import json

# initialize the json object
i_string = """ {'C_code': 1, 'C++_code' : 26,  
      'Java_code' : 17, 'Python_code' : 28}  
"""

# printing initial json
print("The declared dictionary is ", i_string)
print("Its type is ", type(i_string))

# converting string to json
res_dictionary = eval(i_string)

# printing the final result
print("The resultant dictionary is ", str(res_dictionary))
print("The type of resultant dictionary is ", type(res_dictionary))
Output:

The
declared
dictionary is {'C_code': 1, 'C++_code': 26,
               'Java_code': 17, 'Python_code': 28}

Its
type is <


class 'str'>


The
resultant
dictionary is {'C_code': 1, 'C++_code': 26, 'Java_code': 17, 'Python_code': 28}
The
type
of
resultant
dictionary is <


class 'dict'>


Explanation:

Let
us
understand
what
we
have
done in the
above
program.

Since
here
the
objective is to
convert
a
string
to
json
file
we
will
first
import the

json
module.
The
next
step is to
initialize
the
json
object in which
we
have
the
subject
name as keys and then
their
corresponding
values
are
specified.
After
this, we
have
used
eval()
to
convert
a
Python
string
to
json.
On
executing
the
program, it
displays
the
desired
output.

Fetching
values
Finally, in the
last
program
we
will
fetch
the
values
after
the
conversion
of
string
to
json.

Let
's have a look at it.

import json

i_dict = '{"C_code": 1, "C++_code" : 26, "Java_code":17, "Python_code":28}'
res = json.loads(i_dict)
print(res["C_code"])
print(res["Java_code"])
Output:

1
17
We
can
observe
the
following
things in the
output -

We
have
converted
the
string
to
json
using
json.loads().
After
this
we
have
used
the
keys
"C_code" & "Java_code"
to
fetch
their
corresponding
values.
Conclusion
In
this
Tutorial, we
learned
how
to
convert
a
string
to
json
using
Python.

1.
Numpy
library: We
should
make
sure
that
the
numpy
library is installed in our
system and that
too
of
the
latest
version as we
are
going
to
use
functions
on
the
numpy
library
on
the
dataset
we
will
use in the
implementation
process.If
numpy
library is not present in our
system or we
haven
't installed it before, then we can use the following command in the command prompt terminal present in our device to install it:

pip
install
numpy

----------sa
cmd
to

When
we
press
the
enter
key, the
numpy
library is started
installing in our
system.

After
some
time, we
will
see
that
the
numpy
library is successfully
installed in our
system(Here, we
already
have
the
numpy
library
present in our
system).

2.
Panda
library: Like
numpy
library, panda
library is also
the
required
library
that
should
be
present in our
system, and if it is not present in our system, we can use the following command in the command prompt terminal to install it with pip installer:

pip
install
pandas
3.
matplotlib
library: It is also
an
important
library in the
implementation
process
of
the
DBSCAN
algorithm as functions
of
this
library
will
help
us
display
results
from the dataset.If
the
matplotlib
library is not present in our
system, then
we
can
use
the
following
command in the
command
prompt
terminal
present
to
install
it
with pip installer:

pip
install
matplotlib
4.
Sklearn
library: Sklearn
library is going
to
be
one
of
the
major
requirements
while performing the implementation operation of the DBSCAN algorithm as we have to import various modules from the Sklearn library itself in the program, such as preprocessing decomposing etc.Therefore, we should make sure that the Sklearn library is present in our system or not, and if it is not present in our system, then we can use the following command in the command prompt terminal present to install it with pip installer:

pip
install
matplotlib
5.
Last
but
not least, we
should
also
be
aware
of
the
DBSCAN
algorithm(what
it is and how
it
works), as we
have
discussed
already, so
that
we
can
easily
understand
the
implementation
of
it in Python.

Before
we
move
forward, we
should
make
sure
that
we
have
fulfilled
all
the
prerequisites
that
we
have
listed
down
above
so
that
we
don
't have to face any problems while following the implementation steps.

Implementation
steps
for the DBSCAN algorithm:
    Now, we
    will
    perform
    the
    implementation
    of
    the
    DBSCAN
    algorithm in Python.Still, we
    will
    do
    this in steps as we
    have
    mentioned
    earlier
    so
    that
    the
    implementation
    part
    does
    not get
    any
    complex, and we
    can
    understand
    it
    very
    easily.We
    have
    to
    follow
    the
    following
    steps in order
    to
    implement
    the
    DBSCAN
    algorithm and its
    logic
    inside
    a
    Python
    program:

Step
1: Importing
all
the
required
libraries:

First and foremost, we
have
to
import all

the
required
libraries
which
we
have
installed in the
prerequisites
part
so
that
we
can
use
their
functions
while implementing the DBSCAN algorithm.

Here, we
have
firstly
imported
all
the
required
libraries or modules
of
libraries
inside
the
program:

# Importing numpy library as nmp
import numpy as nmp
# Importing pandas library as pds
import pandas as pds
# Importing matplotlib library as pplt
import matplotlib.pyplot as pplt
# Importing DBSCAN from cluster module of Sklearn library
from sklearn.cluster import DBSCAN
# Importing StandardSclaer and normalize from preprocessing module of Sklearn library
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
# Importing PCA from decomposition module of Sklearn
from sklearn.decomposition import PCA

Step
2: Loading
the
Data:

In
this
step, we
have
to
load
that
data, and we
can
do
this
by
importing or loading
the
dataset(that is required in the
DBSCAN
algorithm
to
work
on
it) inside
the
program.To
load
the
dataset
inside
the
program, we
will
use
the
read.csv()
function
of
the
panda
's library and print the information from the dataset as we have done below:

# Loading the data inside an initialized variable
M = pds.read_csv('sampleDataset.csv')  # Path of dataset file
# Dropping the CUST_ID column from the dataset with drop() function
M = M.drop('CUST_ID', axis=1)
# Using fillna() function to handle missing values
M.fillna(method='ffill', inplace=True)
# Printing dataset head in output
print(M.head())

Output

Step
3: Preprocessing
the
data:

Now, we
will
start
preprocessing
the
data
of
the
dataset in this
step
by
using
the
functions
of
preprocessing
module
of
the
Sklearn
library.We
have
to
use
the
following
technique
while preprocessing the data with Sklearn library functions:

# Initializing a variable with the StandardSclaer() function
scalerFD = StandardScaler()
# Transforming the data of dataset with Scaler
M_scaled = scalerFD.fit_transform(M)
# To make sure that data will follow gaussian distribution
# We will normalize the scaled data with normalize() function
M_normalized = normalize(M_scaled)
# Now we will convert numpy arrays in the dataset into dataframes of panda
M_normalized = pds.DataFrame(M_normalized)
Step
4: Reduce
the
dimensionality
of
the
data:

In
this
step, we
will
be
reducing
the
dimensionality
of
the
scaled and normalized
data
so
that
the
data
can
be
visualized
easily
inside
the
program.We
have
to
use
the
PCA
function in the
following
way in order
to
transform
the
data and reduce
its
dimensionality:

# Initializing a variable with the PCA() function
pcaFD = PCA(n_components=2)  # components of data
# Transforming the normalized data with PCA
M_principal = pcaFD.fit_transform(M_normalized)
# Making dataframes from the transformed data
M_principal = pds.DataFrame(M_principal)
# Creating two columns in the transformed data
M_principal.columns = ['C1', 'C2']
# Printing the head of the transformed data
print(M_principal.head())
Output:

As
we
can
see in the
output, we
have
transformed
the
normalized
data
into
two
components
which is the
two
columns(we
can
see
them in the
output), using
the
PCA.And, after
that, we
made
dataframes
from transformed data

using
the
panda
library
dataframe()
function.

Step
5: Build
a
clustering
model:

Now, this is the
most
important
step
of
the
implementation as here
we
have
to
build
a
clustering
model
of
the
data(on
which
we
are
performing
operations), and we
can
do
this
by
using
the
DBSCAN
function
of
the
Sklearn
library as we
have
used
below:

# Creating clustering model of the data using the DBSCAN function and providing parameters in it
db_default = DBSCAN(eps=0.0375, min_samples=3).fit(M_principal)
# Labelling the clusters we have created in the dataset
labeling = db_default.labels_
Step
6: Visualize
the
clustering
model:

# Visualization of clustering model by giving different colours
colours = {}
# First colour in visualization is green
colours[0] = 'g'
# Second colour in visualization is black
colours[1] = 'k'
# Third colour in visualization is red
colours[2] = 'r'
# Last colour in visualization is blue
colours[-1] = 'b'
# Creating a colour vector for each data point in the dataset cluster
cvec = [colours[label] for label in labeling]
# Construction of the legend
# Scattering of green colour
g = pplt.scatter(M_principal['C1'], M_principal['C2'], color='g');
# Scattering of black colour
k = pplt.scatter(M_principal['C1'], M_principal['C2'], color='k');
# Scattering of red colour
r = pplt.scatter(M_principal['C1'], M_principal['C2'], color='r');
# Scattering of green colour
b = pplt.scatter(M_principal['C1'], M_principal['C2'], color='b');
# Plotting C1 column on the X-Axis and C2 on the Y-Axis
# Fitting the size of the figure with figure function
pplt.figure(figsize=(9, 9))
# Scattering the data points in the Visualization graph
pplt.scatter(M_principal['C1'], M_principal['C2'], c=cvec)
# Building the legend with the coloured data points and labelled
pplt.legend((g, k, r, b), ('Label M.0', 'Label M.1', 'Label M.2', 'Label M.-1'))
# Showing Visualization in the output
pplt.show()
Output:

As
we
can
see in the
output, we
have
plotted
the
graph
using
the
data
points
of
the
dataset and visualized
the
clustering
by
labelling
the
data
points
with different colours.

Step
7: Tuning
the
parameters:

In
this
step, we
will
be
tuning
the
parameters
of
the
module
by
changing
the
parameters
that
we
have
previously
given in the
DBSCAN
function as follow:

# Tuning the parameters of the model inside the DBSCAN function
dts = DBSCAN(eps=0.0375, min_samples=50).fit(M_principal)
# Labelling the clusters of data points
labeling = dts.labels_
Step
8: Visualization
of
the
changes:

Now, after
tuning
the
parameters
of
the
cluster
model
we
created, we
will
visualize
the
changes
that
will
come in the
cluster
by
labelling
the
data
points in the
dataset
with different colours as we have done before.

# Labelling with different colours
colours1 = {}
# labelling with Red colour
colours1[0] = 'r'
# labelling with Green colour
colours1[1] = 'g'
# labelling with Blue colour
colours1[2] = 'b'
colours1[3] = 'c'
# labelling with Yellow colour
colours1[4] = 'y'
# Magenta colour
colours1[5] = 'm'
# labelling with Black colour
colours1[-1] = 'k'
# Labelling the data points with the colour variable we have defined
cvec = [colours1[label] for label in labeling]
# Defining all colour that we will use
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
# Scattering the colours onto the data points
r = pplt.scatter(
    M_principal['C1'], M_principal['C2'], marker='o', color=colors[0])
g = pplt.scatter(
    M_principal['C1'], M_principal['C2'], marker='o', color=colors[1])
b = pplt.scatter(
    M_principal['C1'], M_principal['C2'], marker='o', color=colors[2])
c = pplt.scatter(
    M_principal['C1'], M_principal['C2'], marker='o', color=colors[3])
y = pplt.scatter(
    M_principal['C1'], M_principal['C2'], marker='o', color=colors[4])
m = pplt.scatter(
    M_principal['C1'], M_principal['C2'], marker='o', color=colors[5])
k = pplt.scatter(
    M_principal['C1'], M_principal['C2'], marker='o', color=colors[6])
# Fitting the size of the figure with figure function
pplt.figure(figsize=(9, 9))
# Scattering column 1 into X-axis and column 2 into y-axis
pplt.scatter(M_principal['C1'], M_principal['C2'], c=cvec)
# Constructing a legend with the colours we have defined
pplt.legend((r, g, b, c, y, m, k),
            ('Label M.0', 'Label M.1', 'Label M.2', 'Label M.3', 'Label M.4', 'Label M.5', 'Label M.-1'),
            # Using different labels for data points
            scatterpoints=1,  # Defining the scatter point
            loc='upper left',  # Location of cluster scattering
            ncol=3,  # Number of columns
            fontsize=10)  # Size of the font
# Displaying the visualisation of changes in cluster scattering
pplt.show()
Output:

We
can
clearly
observe
the
changes
that
have
come in the
cluster
scattering
of
data
points
by
tuning
the
parameters
of
the
DBSCAN
function
by
looking
at
the
output.As
we
will
observe
the
changes, we
can
also
understand
how
the
DBSCAN
algorithm
works and how
it is helpful in the
Visualization
of
cluster
scattering
of
data
points
present in a
dataset.

--------------cmd
din
to

inspect.getclasstree(classes, unique=False)

inspect.getclasstree(): is used
for arranging the given list of classes into a hierarchy of the nested lists.Where the nested list appears, it will contain the classes derived from the class whose entry would immediately proceed the list.

Example:

# For printing the hierarchy for inbuilt exceptions:
# First, we will import the inspect module
import inspect as ipt


# Then we will create tree_class function
def tree_class(cls, ind=0):
    # Then we will print the name of the class
    print('-' * ind, cls.__name__)

    # now, we will iterate through the subclasses
    for K in cls.__subclasses__():
        tree_class(K, ind + 3)


print("The Hierarchy for inbuilt exceptions is: ")

# THE inspect.getmro() will return the tuple
# of class  which is cls's base classes.

# Now, we will build a tree hierarchy
ipt.getclasstree(ipt.getmro(BaseException))

# function call
tree_class(BaseException)

data
set

tep
1: We
will
import the

libraries.

import numpy as nmp
import matplotlib.pyplot as mpltl
import pandas as pnd

Step
2: We
will
import the

dataset(wine.csv)

First, we
will
import the

dataset and distribute
it
into
X and Y
components
for data analysis.

DS = pnd.read_csv('Wine.csv')

# Now, we will distribute the dataset into two components "X" and "Y"

X = DS.iloc[:, 0:13].values
Y = DS.iloc[:, 13].values
Step
3: In
this
step, we
will
split
the
dataset
into
the
training
set and testing
set.

from sklearn.model_selection import train_test_split as tts

X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=0)
Step
4: Now, we
will
Feature
Scaling.

In
this
step, we
will
do
the
re - processing
on
the
training and testing
set,
for example, fitting the standard scale.

from sklearn.preprocessing import StandardScaler as SS

SC = SS()

X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)
Step
5: Then, Apply
the
PCA
function

We
will
apply
the
PCA
function
into
the
training
set and testing
set
for analysis.

from sklearn.decomposition import PCA

PCa = PCA(n_components=1)

X_train = PCa.fit_transform(X_train)
X_test = PCa.transform(X_test)

explained_variance = PCa.explained_variance_ratio_
Step
6: Now, we
will
fit
Logistic
Regression
for the training set

from sklearn.linear_model import LogisticRegression as LR

classifier_1 = LR(random_state=0)
classifier_1.fit(X_train, Y_train)

Step
7: Here, we
will
predict
the
testing
set
result:

Y_pred = classifier_1.predict(X_test)
Step
8: We
will
create
the
confusion
matrix.

from sklearn.metrics import confusion_matrix as CM

c_m = CM(Y_test, Y_pred)
Step
9: Then, predict
the
result
of
the
training
set.

from matplotlib.colors import ListedColormap as LCM

X_set, Y_set = X_train, Y_train
X_1, X_2 = nmp.meshgrid(nmp.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                        nmp.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

mpltl.contourf(X_1, X_2, classifier_1.predict(nmp.array([X_1.ravel(),
                                                         X_2.ravel()]).T).reshape(X_1.shape), alpha=0.75,
               cmap=LCM(('yellow', 'grey', 'green')))

mpltl.xlim(X_1.min(), X_1.max())
mpltl.ylim(X_2.min(), X_2.max())

for s, t in enumerate(nmp.unique(Y_set)):
    mpltl.scatter(X_set[Y_set == t, 0], X_set[Y_set == t, 1],
                  c=LCM(('red', 'green', 'blue'))(s), label=t)

mpltl.title('Logistic Regression for Training set: ')
mpltl.xlabel('PC_1')  # for X_label
mpltl.ylabel('PC_2')  # for Y_label
mpltl.legend()  # for showing legend

# show scatter plot
mpltl.show()
Output:

Step
10: At
last, we
will
visualize
the
result
of
the
testing
set.

from matplotlib.colors import ListedColormap as LCM

X_set, Y_set = X_test, Y_test

X_1, X_2 = nmp.meshgrid(nmp.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                        nmp.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

mpltl.contourf(X_1, X_2, classifier_1.predict(nmp.array([X_1.ravel(),
                                                         X_2.ravel()]).T).reshape(X_1.shape), alpha=0.75,
               cmap=LCM(('pink', 'grey', 'aquamarine')))

mpltl.xlim(X_1.min(), X_1.max())
mpltl.ylim(X_2.min(), X_2.max())

for s, t in enumerate(nmp.unique(Y_set)):
    mpltl.scatter(X_set[Y_set == t, 0], X_set[Y_set == t, 1],
                  c=LCM(('red', 'green', 'blue'))(s), label=t)

# title for scatter plot
mpltl.title('Logistic Regression for Testing set')
mpltl.xlabel('PC_1')  # for X_label
mpltl.ylabel('PC_2')  # for Y_label
mpltl.legend()

# show scatter plot
mpltl.show()
Output:

Suppose
we
have
given
two
dates
our
expected
output
would
be:

Example:

Input: Date_1 = 12 / 10 / 2021, Date_2 = 31 / 0
8 / 2022
Output: Number
of
Days
between
the
given
Dates
are: 323
days
Input: Date_1 = 10 / 0
9 / 2023, Date_2 = 04 / 02 / 2025
Output: Number
of
Days
between
the
given
Dates
are: 323
days: 513
days

Example:


# First, we will create a class for dates
class date_n:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year

    # For storng number of days in all months from


# January to December.
month_Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# This function will count the number of leap years from 00/00/0000 to the #given date


def count_Leap_Years(day):
    years = day.year

    # Now, it will check if the current year should be considered for the count          # of leap years or not.
    if (day.month <= 2):
        years -= 1

        # The condition for an year is a leap year: if te year is a multiple of 4, and a            # multiple of 400 but not a multiple of 100.
    return int(years / 4) - int(years / 100) + int(years / 400)


# This function will return number of days between two given dates
def get_difference(date_1, date_2):
    # Now, it will count total number of days before first date "date_1"

    # Then, it will initialize the count by using years and day
    n_1 = date_1.year * 365 + date_1.day

    # then, it will add days for months in the given date
    for K in range(0, date_1.month - 1):
        n_1 += month_Days[K]

        # As every leap year is of 366 days, we will add
    # a day for every leap year
    n_1 += count_Leap_Years(date_1)

    # SIMILARLY, it will count total number of days before second date "date_2"

    n_2 = date_2.year * 365 + date_2.day
    for K in range(0, date_2.month - 1):
        n_2 += month_Days[K]
    n_2 += count_Leap_Years(date_2)

    # Then, it will return the difference between two counts
    return (n_2 - n_1)


# Driver program
date_1 = date_n(12, 10, 2021)
date_2 = date_n(30, 8, 2022)

print("Number of Days between the given Dates are: ", get_difference(date_1, date_2), "days")
Output:

The
installation
command
for the same is shown below:

Syntax:

# installing OpenCV
$ pip
install
opencv - python

# installing TensorFlow
$ pip
install
tensorflow

# installing Keras
$ pip
install
keras

# installing ImageAI
$ pip
install
imageAI

Step
1
The
initial
step is to
create
the
necessary
folders.For
this
tutorial, we
will
need
the
folders as shown
below:

Object_Recognition: This
will
be
the
root
folder.
Models: This
folder
will
store
the
pre - trained
model.
Input: This
folder
will
store
the
image
file
on
which
we
have
to
perform
object
detection.
Output: This
folder
will
store
the
image
file
with detected objects.
Once
we
created
the
necessary
folder, the
Object
Recognition
folder
should
have
the
following
sub - folders:

?   Input
?   Models
?   Output
3
directories, 0
files
Step
2
For
the
second
step, we
will
open
the
preferred
text
editor, which is Visual
Studio
Code, in this
case, to
write
a
Python
script and create
a
new
file
recognizer.py

Step
3
Now, let
us
begin
importing
ObjectDetection


class from the ImageAI library.The syntax for the same is shown below:


File: recognizer.py

# importing the required library
from imageai.Detection import ObjectDetection

Step
4
Now
that
the
required
ImageAI
library is imported and the
ObjectDetection


class , the next thing is to create an instance of the class ObjectDetection.Let us consider the following snippet of code for the same.


File: recognizer.py

# instantiating the class
recognizer = ObjectDetection()
Step
5
Let
us
specify
the
path
from the model, input

image, and output
image
using
the
following
snippet
of
code.

File: recognizer.py

# defining the paths
path_model = "./Models/yolo-tiny.h5"
path_input = "./Input/images.jpg"
path_output = "./Output/newimage.jpg"
Step
6
Once, we
instantiated
the
ObjectDetection


class we can now call different functions from the class.The class consists of the following functions in order to call pre-trained models:


setModelTypeAsRetinaNet()
setModelTypeAsYOLOv3()
setModelTypeAsTinyYOLOv3()
For
this
tutorial
's purpose, we will utilize the pre-trained TinyYOLOv3 model, and thus, we will be using the setModelTypeAsTinyYOLOv3() function in order to load the model.

Let
us
consider
the
following
snippet
of
code
for the same:

File: recognizer.py

# using the setModelTypeAsTinyYOLOv3() function
recognizer.setModelTypeAsTinyYOLOv3()
Step
7
Now, we
will
be
going
to
call
the
function
setModelPath().This
function
will
accept
a
string
that
consists
of
the
path
to
the
pre - trained
model.

Let
us
consider
the
following
snippet
of
code
for the same:

File: recognizer.py

# setting the path to the pre-trained Model
recognizer.setModelPath(path_model)
Step
8
In
this
step, we
will
call
the
loadModel()
function
from the recognizer

instance.This
function
will
load
the
model
from the path

specified
above
with the help of the setModelPath() class method.

Let
us
consider
the
following
snippet
of
code
for the same.

File: recognizer.py

# loading the model
recognizer.loadModel()
Step
9
We
have
to
call
the
detectObjectsFromImage()
function
with the help of the recognizer object that we created earlier.

This
function
accepts
two
parameters: input_image and output_image_path.The
input_image
parameter is the
path
where
the
image
we
recognise is situated, whereas
the
output_image_path
parameter is the
path
storing
the
image
with detected objects.This function will return a diction containing the names and percentage probabilities of every object detected in the image.

The
syntax
for the same is shown below:

File: recognizer.py

# calling the detectObjectsFromImage() function
recognition = recognizer.detectObjectsFromImage(
    input_image=path_input,
    output_image_path=path_output
)
Step
10
At
last, we
can
access
the
dictionary
elements
by
iterating
through
each
element
present in the
dictionary.

The
syntax
for the same is shown below:

File: recognizer.py

# iterating through the items found in the image
for eachItem in recognition:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])

--------bagong
save
recognizer.py

Let
us
consider
the
following
script
for the Object Recognition model.

File: recognizer.py

# importing the required library
from imageai.Detection import ObjectDetection

# instantiating the class
recognizer = ObjectDetection()

# defining the paths
path_model = "./Models/yolo-tiny.h5"
path_input = "./Input/images.jpg"
path_output = "./Output/newimage.jpg"

# using the setModelTypeAsTinyYOLOv3() function
recognizer.setModelTypeAsTinyYOLOv3()
# setting the path of the Model
recognizer.setModelPath(path_model)
# loading the model
recognizer.loadModel()
# calling the detectObjectsFromImage() function
recognition = recognizer.detectObjectsFromImage(
    input_image=path_input,
    output_image_path=path_output
)

# iterating through the items found in the image
for eachItem in recognition:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])
Output:

----------eto
actual
image

eto
na
yung
imageAI
kung
tawagin
yung
may
scan

Fixing
errors
that
may
occur
while importing the VLC module
In
case
the
path is not defined, the
issue is that
dll is not in the
PATH(System
Variable).We
have
to
include
the
libvlc.dll
file
path
to
the
system
variable
to
solve
the
problem.We
can
find
this
file in the
VLC
folder
where
it is installed.
Wrong
version
of
VLC: Generally, users
install
32
bits
version
of
VLC, which
may
cause
some
trouble if we
have
installed
the
64
bits
version
of
Python.All
we
have
to
reinstall
the
64
bits
version
of
VLC in order
to
fix
this.
We
can
import the

OS
module
before
the
VLC
module and register
dll
using
the
following
syntax:
Syntax:

os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')
Some
examples
of
the
VLC
module
Let
us
consider
a
simple
program
to
play
Video
using
VLC.

Example:

# importing the vlc module
import vlc

# creating the vlc media player object
my_media = vlc.MediaPlayer("video.mp4")

# playing video
my_media.play()
Explanation:

In
the
above
snippet
of
code, we
have
imported
the
VLC
module.We
have
then
created
an
object
of
the
VLC
media
player.At
last, we
have
used
the
play()
function in order
to
play
the
video.

Now, let
us
consider
another
example
deriving
the
duration
of
a
video
file
using
the
VLC
module.

Example:

# importing the time and vlc modules
import time, vlc


# defining the method to play video
def vlc_video(src):
    # creating an instance of vlc
    vlc_obj = vlc.Instance()

    # creating a media player
    vlcplayer = vlc_obj.media_player_new()

    # creating a media
    vlcmedia = vlc_obj.media_new(src)

    # setting media to the player
    vlcplayer.set_media(vlcmedia)

    # playing the video
    vlcplayer.play()

    # waiting time
    time.sleep(0.5)

    # getting the duration of the video
    video_duration = vlcplayer.get_length()

    # printing the duration of the video
    print("Duration : " + str(video_duration))


# calling the video method
vlc_video("video.mp4")

Set
to
list in Python
In
this
article, we
will
discuss
how
we
can
convert
a
set
to
a
list in Python.

Before
that
let
's have a quick revision of lists and sets.

List - It is a
sequence
of
elements
enclosed in square
brackets
where
each
element is separated
with a comma.

Syntax
of
a
list is -

a = [1, 2, 4.5, 'Python', 'Java']
We
can
print
the
list and check
its
type
using -

print(a)
print(type(a))
NOTE: The
list is mutable
which
means
we
can
change
its
elements.
Set - It is an
unordered
collection
of
elements
that
contains
all
the
unique
values
enclosed
within
curly
brackets.

Syntax
of
a
set is -

b = {1, 2, 4.5, 'Python', 'Java'}
We
can
print
the
set and check
its
type
using -

print(b)
print(type(b))
The
different
approaches
of
converting
a
set
to
a
string
that
we
will
use
are -

Using
list()
Using
sorted()
Using * set
Using
for loop
    Using
    frozenset
Using
list()
In
the
first
method, we
will
use
list()
to
convert
the
set.

The
following
program
shows
how
it
can
be
done -

# declaring a set
subjects = {'C', 'C++', 'Java', 'Python', 'HTML'}
# using list()
res = list(subjects)
print(res)

Explanation:

Let
us
understand
what
we
have
done in the
above
program -

The
first
thing
that
we
have
done
here is to
declare
the
set
that
consists
of
different
subject
names.
After
this, we
have
used
list()
function in which
we
passed
the
set
'subjects'.
On
executing
the
program, the
desired
output is displayed.
Using
sorted()
The
second
approach is to
use
the
sorted()
function
to
convert
a
set
to
a
list.

The
program
below
illustrates
the
same -


# defining a function
def convert_set(set):
    return sorted(set)


subjects = {'C', 'C++', 'Java', 'Python', 'HTML'}
res = set(subjects)
print(convert_set(res))

Explanation:

Let
us
understand
what
we
have
done in the
above
program -

The
first
thing
that
we
have
done
here is, we
created
a
function
that
takes
a
set as its
parameter and returns
the
expected
output.
After
this, we
have
declared
the
variable
of
a
set
type
that
consists
of
different
subject
names.
The
next
step
was
to
pass
our
set in the
function
'convert_set'.
On
executing
the
program, the
desired
output is displayed.
Using * set
In
the
third
method, we
will
use
the * set
to
convert
a
set
to
a
list in Python.

The * set
unpacks
the
set
inside
a
list.

The
following
program
shows
how
it
can
be
done -


# defining a function
def convert_set(set):
    return [*set, ]


res = set({'C', 'C++', 'Java', 'Python', 'HTML'})
print(convert_set(res))
Explanation:

Let
us
understand
what
we
have
done in the
above
program -

The
first
thing
that
we
have
done
here is, we
created
a
function
that
takes
a
set as its
parameter and returns
the
expected
output.
After
this, we
have
passed
the
value
of
the
set
that
consists
of
different
subject
names
inside
the
set().
The
next
step
was
to
pass
our
set in the
function
'convert_set'.
On
executing
the
program, the
desired
output is displayed.
Output

['C', 'C++', 'Java', 'Python', 'HTML']
Using
for loop
    In
    the
    fourth
    method, we
    will
    use
    for loop to convert a set to a list in Python.

The
program
below
illustrates
the
same -

# using for loop
subjects = set({'C', 'C++', 'Java', 'Python', 'HTML'})

res = []

for i in subjects:
    res.append(i)
Output:

['C', 'C++', 'Java', 'Python', 'HTML']
Let
us
understand
what
we
have
done in the
above
program -

The
first
thing
that
we
have
done
here is to
declare
the
set
that
consists
of
different
subject
names.
After
this, we
have
declared
an
empty
list
res.
We
have
used
for loop here, that took each element from the set and added it to the list.
On
executing
the
program, the
desired
output is displayed.
Using
frozenset
Finally, in the
last
method, we
will
use
frozenset
to
convert
a
set
to
a
list in Python.

The
difference
between
a
set and a
frozenset is that
a
set is mutable
whereas
a
frozenset is immutable.

The
following
program
shows
how
it
can
be
done -

subjects = frozenset({'C', 'C++', 'Java', 'Python', 'HTML'})

res = list(subjects)

print(res)
Output:

['C', 'C++', 'Java', 'Python', 'HTML']
Explanation:

Let
us
understand
what
we
have
done in the
above
program -

The
first
thing
that
we
have
done
here is to
declare
the
frozenset
that
consists
of
different
subject
names.
After
this, we
have
used
list() in which
we
passed
the
set
'subjects'.
On
executing
the
program, the
desired
output is displayed.

a = 'Learning Python is fun'
b = 20
# Displaying the type of a and b
print(type(a))
print(type(b))
Output:

<

class 'str'>

<

class 'int'>


In
the
above
example, we
have
declared
the
variables
'a' and 'b'
with a string and an integer value respectively.
Play
Videox

We
can
verify
their
data
types
using
type().

The
question
that
arises
here is why
do
we
need
to
convert
a
string
to
an
integer.

The
following
program
illustrates
the
same -

value_a = "100"
value_b = "26"
res = value_a * value_b
print("The multiplication of val_a and val_b gives: ", res)
Output:

res = value_a * value_b

TypeError: can
't multiply sequence by non-int of type '
str
'
Since
it
generates
this
type
of
error, this is the
reason
that
we
must
convert
the
string
values
to
integers
so
that
we
can
easily
proceed
with the operations.

It
's time to have a look at the first program that demonstrates converting a string to an integer.

a = '7'
print(type(a))
# using int()
conv_a = int(a)
print(type(conv_a))
conv_a = conv_a + 10
print(conv_a)
print(type(conv_a))
Output:

<

class 'str'>

<

class 'int'>


17
<

class 'int'>


Explanation:

Let
's see the explanation of the above program-

The
first
step is to
declare
the
variable
'a'
with a string value.
After
this, we
have
checked
its
data
type
using
type().
For
converting
the
string
to
an
integer, we
have
used
int() and then
checked
its
type.
Now
we
have
operated
on
the
variable
'a'
by
adding
10
to
it.
Finally, the
resultant
value is displayed in the
output.
Approach - 2
In
the
next
example, we
will
go
for an indirect approach of converting a string to an integer.

The
following
program
shows
how
it
can
be
done -

value_a = "100"
value_b = "26"
print(type(value_a))
print(type(value_b))
# converting to float
value_a = float(value_a)
# converting to int
value_b = int(value_b)
res_sum = value_a + value_b
print("The sum of value_a and value_b is ", res_sum)
Output:

<

class 'str'>

<

class 'str'>


The
sum
of
value_a and value_b is 126.0
Explanation:

Let
us
understand
what
we
have
done in the
above
program -

The
first
step is to
declare
the
two
variables
'value_a' and 'value_b'
with a string value.
After
this, we
have
checked
their
data
type
using
type().
For
converting
the
string
to
an
integer, we
have
used
float()
to
convert
the
string
to
float
value.
In
the
next
step, we
will
convert
the
string
value
of
'value_b'
to
an
integer.
Now
we
have
added
'value_a' and 'value_b' and printed
their
sum.
Finally, the
resultant
value is displayed in the
output.
Approach - 3:
In
the
last
program, we
will
discuss
one
more
scenario
of
converting
string
to
int in Python.

Here
we
will
see
how
we
can
convert
a
number
present as a
string
value
to
base
10
when
it is on
different
bases.

The
following
program
illustrates
the
same -

num_value = '234'
# printing the value of num_value
print('The value of num_value is :', num_value)
# converting 234 to base 10 assuming it is in base 10
print('The value of num_value from base 10 to base 10 is:', int(num_value))
# converting 234 to base 10 assuming it is in base 8
print('The value of num_value from base 8 to base 10 is :', int(num_value, base=8))
# converting 234 to base 10 assuming it is in base 6
print('The value of num_value base 6 to base 10 is :', int(num_value, base=6))
Output:

The
value
of
num_value is: 234
The
value
of
num_value
from base

10
to
base
10 is: 234
The
value
of
num_value
from base

8
to
base
10 is: 156
The
value
of
num_value
base
6
to
base
10 is: 94
Explanation:

It
's time to have a glance at the explanation of the above program.

In
the
first
step, we
have
declared
the
value
of
the
variable.
Since
the
output
will
always
be in base
10, we
have
provided
the
different
base
values
inside
int().
The
base
values
we
have
taken
here
are
10, 8, and 6.
On
executing
the
program, the
expected
output is displayed.

# importing the required modules
from gpiozero import Button
from time import sleep

# creating an object of Button
the_button = Button(2)

# using the if-else statement
while True:
    if the_button.is_pressed:
        print("Button Pressed")
    else:
        print("Button Released")
    sleep(1)
Explanation:

The
above
example
demonstrates
the
receiving and processing
of
the
signals
by
pressing
the
button
on
the
second
pin
at
the
moment
of
release.

Syntax:

$ pip
install
esptool

# importing the required modules
from machine import Pin
import time

# creating an object of Pin
ledPin = Pin(2, Pin.OUT)

# using some functions
while True:
    ledPin.on()
    time.sleep(1)
    ledPin.off()
    time.sleep(1)
Explanation:

In
the
above
snippet
of
code, we
have
imported
the
Pin
module
from the machine

library
along
with the time module.We have then created an object of Pin and execute some functions on it.

How
to
install
the
pysftp
module?
The
pysftp
interface
doesn
't expose most of the features of Paramiko; however, it abstracts pretty much features using a single method. In contrast, the pysftp module implements more high-level features on top of Paramiko, notably recursive file transfers.

We
can
install
the
pysftp
module
with the help of the pip installer using the following command:

Syntax:

$ pip
install
pysftp
# or
$ python3 - m
pip
install
pysftp
The
module
will
be
installed in the
system as the
version
of
Python and pip.

Verifying
the
Installation
In
order
to
check
whether
the
module
has
been
installed in the
system
properly or not, we
can
try importing the module and execute the program.

Once
the
installation is complete, create
a
new
Python
file and type
the
following
syntax in it.

Example:

# importing the required module
import pysftp

Now, save
the
file and run
the
file
using
the
following
command in the
command
prompt.

Syntax:

$ python < file - name >.py
If
the
program
runs
without
raising
any
import error, the

module is installed
properly.Else
it is recommended
to
reinstall
the
module and refer
to
its
official
documentation.

Accessing
SFTP
Server
using
pysftp
We
can
list
the
content
of
the
direction
with the help of the pysftp module in Python.In order to accomplish the goal, we need the hostname, username, and password.

Then
we
have
to
switch
from direction using

either
the
chdir or cwd
method and provide
the
first
argument as the
remote
directory.

Let
us
consider
the
following
example
for the same.

# importing the required module
import pysftp

# defining the host, username, and password
my_Hostname = "welcomeblog.com"
my_Username = "root"
my_Password = "root"
# using the python with statement
with pysftp.Connection(
        host=my_Hostname,
        username=my_Username,
        password=my_Password
) as sftp:
    print("Connection succesfully established ... ")
    # Switching to a remote directory
    sftp.cwd('/var/www/vhosts/')
    # Obtaining structure of the remote directory '/var/www/vhosts'
    dir_struct = sftp.listdir_attr()
    # Printing data
    for attr in dir_struct:
        print(attr.filename, attr)
    # connection closed automatically at the end of the with statement
Explanation:

The
above
snippet
of
code is a
dummy
server
that
doesn
't exist. Still, in real life, we have to utilize the environment variables to fetch the original credentials in any file for security purposes and not pull all the credentials in individual files. It is recommended to put it inside the environment variables file. For instance, the .env file.

Now, let
us
understand
the
above
code.The
above
snippet
of
code is the
same
for anyone as we have to provide the credentials, and the program will start working.

First, we
have
imported
the
pysftp
module and then
provided
the
variables
to
store
the
value
of
hostname, username and password.We
have
then
used
Python
with statement to open the secure connection to the remote server by providing hostname, username, and password.If it is successful, we will switch the remote directory, fetch the listing and print one by one in the console.

The
list is in arbitrary
order, and it
doesn
't involve the unique entries '.
' and '..
'. The returned SFTPAttributes objects will have the additional field: longname, which may consist of the formatted string of the attributes of a file in UNIX format. The string'
s
content
will
rely
on
the
SFTP
server.

Uploading
a
file
using
pysftp in Python
We
can
upload
a
file in the
remote
server
through
SFTP
with the help of pysftp using the sftp.put() function of the SFTP client.The put method expects the relative or absolute local path of the file we need to upload as the first argument and the remote path where the file should be uploaded as the second one.

Let
us
consider
the
following
snippet
of
code
for better understanding.

Example:

# importing the required module
import pysftp

# defining the host, username, and password
my_Hostname = "welcomeblog.com"
my_Username = "root"
my_Password = "root"
# using the python with statement
with pysftp.Connection(
        host=my_Hostname,
        username=my_Username,
        password=my_Password
) as sftp:
    print("Connection succesfully established ... ")
    # Defining a file that we want to upload from the local directorty
    # or absolute "/Users/krunal/Desktop/code/pyt/app.txt"
    local_File_Path = './app.txt'
    # Defining the remote path where the file will be uploaded
    remote_File_Path = '/var/backups/app.txt'
    # Using the put method to upload a file
    sftp.put(local_File_Path, remote_File_Path)
# connection closed automatically at the end of the with statement
Explanation:

In
the
above
snippet
of
code, we
have
established
a
secure
connection and then
defined
two
file
paths - local_File_Path and remote_File_Path.

local_File_Path: This
file
path is the
path
to
the
local
file.
remote_File_Path: This
file
path is the
path
to
the
remote
file.
Then, we
have
used
the
sftp.put()
function in order
to
upload
a
file
to
the
server.

Downloading
a
remote
file
using
pysftp in Python
In
the
previous
section, we
have
discussed
the
method
of
uploading
a
file
using
pysftp.Let
us
understand
the
method
of
downloading
a
file.

We
can
download
a
remote
file
from the server

with the help of pysftp by opening a connection and from the sftp instance and utilizing the get method expecting the path of a remote file that will be downloaded.The second parameter is a local path where we should store the file.

Let
us
consider
the
following
example
demonstrating
the
same.

Example:

# importing the required module
import pysftp

# defining the host, username, and password
my_Hostname = "welcomeblog.com"
my_Username = "root"
my_Password = "root"
# using the python with statement
with pysftp.Connection(
        host=my_Hostname,
        username=my_Username,
        password=my_Password)
    as sftp:
    print("Connection succesfully established ... ")
    # Defining the remote path file path
    remote_File_Path = '/var/backups/app.txt'
    # Defining a directory in which we have to save the file.
    # or absolute "/Users/krunal/Desktop/code/pyt/app.txt"
    local_File_Path = './app.txt'
    # Using the get method to download a file
    sftp.get(remote_File_Path, local_File_Path)
# connection closed automatically at the end of the with statement
Explanation:

In
the
above
snippet
of
code, we
have
defined
a
connection and then
defined
two
file
paths - remote_File_Path and local_File_Path

remote_File_Path: This
file
path is a
path
where
the
file is located.
local_File_Path: This
file
path is a
path
where
the
file
will
be
downloaded.
We
have
then
utilized
the
sftp.get()
function in order
to
download
the
file.

Deleting
a
file
using
pysftp in Python
We
can
remove
a
file
with the help of pysftp using the sftp.remove() function.The remove() function expects the absolute path to the remote file as the first parameter.

Let
us
consider
the
following
example
demonstrating
the
same.

Example:

# importing the required module
import pysftp

# defining the host, username, and password
my_Hostname = "welcomeblog.com"
my_Username = "root"
my_Password = "root"
# using the python with statement
with pysftp.Connection(
        host=my_Hostname,
        username=my_Username,
        password=my_Password)
    as sftp:
    print("Connection succesfully established ... ")
    # Defining the remote path file path
    remote_File_Path = '/var/backups/app.txt'
    # Using the get method to download a file
    sftp.remove(remote_File_Path)
# connection closed automatically at the end of the with statement
Explanation:

In
the
above
snippet
of
code, we
have
opened
a
connection and then
defined
a
remove_File_Path
variable, which
consists
of
the
path
of
a
file
that
requires
to
be
deleted.

We
have
then
utilized
the
sftp.remove()
function in order
to
delete
a
file
from the remote

server.

The
pysftp
module
has
a
large
variety
of
functions
that
we
can
utilize
to
perform
various
activities, such as handling
permissions and a
lot
more.One
can
also
refer
to
the
official
documentation
of
the
Python
pysftp
module.

Output
expression
Input
sequence
A
member
of
the
input
sequence
represented
by
a
variable
The
optional
predicate
parts.
Example:

import functools as FT

# First, filter odd numbers
list_1 = filter(lambda K: K % 2 == 1, range(10, 30))
print("List: ", list(list_1))

# Then we will filter the odd square which is divisible by 5
list_1 = filter(lambda K: K % 5 == 0,
                [K ** 2 for K in range(1, 11) if K % 2 == 1])
print("ODD SQUARE WHICH IS DIVISIBLE BY 5: ", list(list_1))

# Here, we will filter negative numbers
list_1 = filter((lambda K: K < 0), range(-10, 10))
print("Filter negative numbers: ", list(list_1))

# Now, implement by using the max() function
print("Maximum Number in the List: ")
print(FT.reduce(lambda S, T: S if (S > T) else T, [14, 11, 65, 110, 105]))

2.
Printing
a
List: Lists
are
not printed
according
to
our
requirements;
they
are
always
printed in unnecessary
square
brackets and single
quotes.But in Python, we
have
a
solution
for printing lists efficiently by using the join method of string.The "join method" can turn the list into a string by classifying every item into a string and connecting them with the string on which the join method is used.

Example:

# First declare the list:
ABC = ['LPG', 'WWF', 'XYZ', 'MPG']

# Then, we will print the list:
print("The Simple List: ", ABC)

# HEre, we will Print the list by using join method
print('The List by using join method: %s' % ', '.join(ABC))

# we can directly apply Join Function on the List:
print('Directly applying the join method: ', (", ".join(ABC)))
Output:

The
Simple
List: ['LPG', 'WWF', 'XYZ', 'MPG']
The
List
by
using
join
method: LPG, WWF, XYZ, MPG
Directly
applying
the
join
method: LPG, WWF, XYZ, MPG
3.
Transpose
a
Matrix: In
Python, a
user
can
implement
the
matrix as a
nested
list, which
means
a
list
inside
a
list.Every
element
of
the
list is treated as a
row
of
the
matrix.

Example:

M_1 = [[5, 3], [1, 2], [9, 8]]
print("Matrix 1: ")
for row in M_1:
    print(row)
rez_1 = [[M_1[K][L] for K in range(len(M_1))] for L in range(len(M_1[0]))]
print("\n")
print("Matrix 2: ")
for row in rez_1:
    print(row)
Output:

Matrix
1:
[5, 3]
[1, 2]
[9, 8]

Matrix
2:
[5, 1, 9]
[3, 2, 8]
4.
artition
of
List
into
"N"
Groups: The
users
can
use
the
iter()
function as an
iterator
over
the
sequence.

Example:

# First, we will Declare the list:
LIST_1 = ['E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'E_6']

partition_1 = list(zip(*[iter(LIST_1)] * 2))
print("List after partitioning into different of groups of two elements: ", partition_1)
Output:

List
after
partitioning
into
different
of
groups
of
two
elements: [('E_1', 'E_2'), ('E_3', 'E_4'), ('E_5', 'E_6')]
Explanation:

In
the
above
code, we
used
"[iter(LIST_1)] * 2"
which
produced
different
groups
containing
two
elements
of
the
'LIST_1[]'
list.That is, the
lists
of
length
two
will
be
generated
using
the
elements
from the first

list.

5.
Print
more
than
One
Item
of
List
simultaneously

Example:

list_1 = [11, 13, 15, 17]
list_2 = [12, 14, 16, 18]

# Here, we will use zip() function which will take 2 equal length list
# and then merge them together into pairs
for K, L in zip(list_1, list_2):
    print(K, L)
Output:

11
12
13
14
15
16
17
18
6.
Take
the
String as Input and Convert
it
into
List:

Example:

# Reading a string from input as int format
# after splitting it's elements by white-spaces
print("Input: ")
formatted_list_1 = list(map(int, input().split()))
print("Output as Formatted list: ", formatted_list_1)
Output:

Input:
10
12
14
16
18
20
22
Output as Formatted
list: [10, 12, 14, 16, 18, 20, 22]
7.
Convert
List
of
Lists
into
Single
List:

Example:

# importing the itertools
import itertools as IT

# Declaring the list geek
LIST_1 = [[1, 2], [3, 4], [5, 6]]

# chain.from_iterable() function will return the
# elements of nested list
# and iterate it from first list
# of iterable till the last
# end of the list

list_2 = list(IT.chain.from_iterable(LIST_1))
print("Iterated list of 'LIST_1': ", list_2)
Output:

Iterated
list
of
'LIST_1': [1, 2, 3, 4, 5, 6]
8.
Print
the
Repeated
Characters: Suppose
our
task is to
print
the
patterns
like
"122333444455555666666".We
can
easily
print
this
pattern in Python
without
using
for loop.

Example:

print("1" + "2" * 2 + "3" * 3 + "4" * 4 + "5" * 5 + "6" * 6)
Output:

122333444455555666666









