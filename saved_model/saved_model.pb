√л
гє
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

└
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718оэ
|
Conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv1/kernel
u
 Conv1/kernel/Read/ReadVariableOpReadVariableOpConv1/kernel*&
_output_shapes
: *
dtype0
l

Conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Conv1/bias
e
Conv1/bias/Read/ReadVariableOpReadVariableOp
Conv1/bias*
_output_shapes
: *
dtype0
|
Conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameConv2/kernel
u
 Conv2/kernel/Read/ReadVariableOpReadVariableOpConv2/kernel*&
_output_shapes
:  *
dtype0
l

Conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Conv2/bias
e
Conv2/bias/Read/ReadVariableOpReadVariableOp
Conv2/bias*
_output_shapes
: *
dtype0
Р
Conv1_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameConv1_transpose/kernel
Й
*Conv1_transpose/kernel/Read/ReadVariableOpReadVariableOpConv1_transpose/kernel*&
_output_shapes
:  *
dtype0
А
Conv1_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameConv1_transpose/bias
y
(Conv1_transpose/bias/Read/ReadVariableOpReadVariableOpConv1_transpose/bias*
_output_shapes
: *
dtype0
Р
Conv2_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameConv2_transpose/kernel
Й
*Conv2_transpose/kernel/Read/ReadVariableOpReadVariableOpConv2_transpose/kernel*&
_output_shapes
:  *
dtype0
А
Conv2_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameConv2_transpose/bias
y
(Conv2_transpose/bias/Read/ReadVariableOpReadVariableOpConv2_transpose/bias*
_output_shapes
: *
dtype0
К
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameoutput_layer/kernel
Г
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*&
_output_shapes
: *
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
К
Adam/Conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/Conv1/kernel/m
Г
'Adam/Conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1/kernel/m*&
_output_shapes
: *
dtype0
z
Adam/Conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Conv1/bias/m
s
%Adam/Conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1/bias/m*
_output_shapes
: *
dtype0
К
Adam/Conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/Conv2/kernel/m
Г
'Adam/Conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2/kernel/m*&
_output_shapes
:  *
dtype0
z
Adam/Conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Conv2/bias/m
s
%Adam/Conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2/bias/m*
_output_shapes
: *
dtype0
Ю
Adam/Conv1_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *.
shared_nameAdam/Conv1_transpose/kernel/m
Ч
1Adam/Conv1_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1_transpose/kernel/m*&
_output_shapes
:  *
dtype0
О
Adam/Conv1_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/Conv1_transpose/bias/m
З
/Adam/Conv1_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1_transpose/bias/m*
_output_shapes
: *
dtype0
Ю
Adam/Conv2_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *.
shared_nameAdam/Conv2_transpose/kernel/m
Ч
1Adam/Conv2_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2_transpose/kernel/m*&
_output_shapes
:  *
dtype0
О
Adam/Conv2_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/Conv2_transpose/bias/m
З
/Adam/Conv2_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv2_transpose/bias/m*
_output_shapes
: *
dtype0
Ш
Adam/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/output_layer/kernel/m
С
.Adam/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/m*&
_output_shapes
: *
dtype0
И
Adam/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/m
Б
,Adam/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/m*
_output_shapes
:*
dtype0
К
Adam/Conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/Conv1/kernel/v
Г
'Adam/Conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1/kernel/v*&
_output_shapes
: *
dtype0
z
Adam/Conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Conv1/bias/v
s
%Adam/Conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1/bias/v*
_output_shapes
: *
dtype0
К
Adam/Conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *$
shared_nameAdam/Conv2/kernel/v
Г
'Adam/Conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2/kernel/v*&
_output_shapes
:  *
dtype0
z
Adam/Conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Conv2/bias/v
s
%Adam/Conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2/bias/v*
_output_shapes
: *
dtype0
Ю
Adam/Conv1_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *.
shared_nameAdam/Conv1_transpose/kernel/v
Ч
1Adam/Conv1_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1_transpose/kernel/v*&
_output_shapes
:  *
dtype0
О
Adam/Conv1_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/Conv1_transpose/bias/v
З
/Adam/Conv1_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1_transpose/bias/v*
_output_shapes
: *
dtype0
Ю
Adam/Conv2_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *.
shared_nameAdam/Conv2_transpose/kernel/v
Ч
1Adam/Conv2_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2_transpose/kernel/v*&
_output_shapes
:  *
dtype0
О
Adam/Conv2_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/Conv2_transpose/bias/v
З
/Adam/Conv2_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv2_transpose/bias/v*
_output_shapes
: *
dtype0
Ш
Adam/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/output_layer/kernel/v
С
.Adam/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/v*&
_output_shapes
: *
dtype0
И
Adam/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/v
Б
,Adam/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
И:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*├9
value╣9B╢9 Bп9
█
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
Ї
5iter

6beta_1

7beta_2
	8decay
9learning_ratemgmhmimj#mk$ml)mm*mn/mo0mpvqvrvsvt#vu$vv)vw*vx/vy0vz
F
0
1
2
3
#4
$5
)6
*7
/8
09
F
0
1
2
3
#4
$5
)6
*7
/8
09
 
н

trainable_variables
:layer_regularization_losses
;metrics

<layers
=non_trainable_variables
>layer_metrics
	variables
regularization_losses
 
XV
VARIABLE_VALUEConv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
trainable_variables
?layer_regularization_losses
@metrics

Alayers
Bnon_trainable_variables
Clayer_metrics
	variables
regularization_losses
 
 
 
н
trainable_variables
Dlayer_regularization_losses
Emetrics

Flayers
Gnon_trainable_variables
Hlayer_metrics
	variables
regularization_losses
XV
VARIABLE_VALUEConv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
trainable_variables
Ilayer_regularization_losses
Jmetrics

Klayers
Lnon_trainable_variables
Mlayer_metrics
	variables
regularization_losses
 
 
 
н
trainable_variables
Nlayer_regularization_losses
Ometrics

Players
Qnon_trainable_variables
Rlayer_metrics
 	variables
!regularization_losses
b`
VARIABLE_VALUEConv1_transpose/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEConv1_transpose/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
н
%trainable_variables
Slayer_regularization_losses
Tmetrics

Ulayers
Vnon_trainable_variables
Wlayer_metrics
&	variables
'regularization_losses
b`
VARIABLE_VALUEConv2_transpose/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEConv2_transpose/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
н
+trainable_variables
Xlayer_regularization_losses
Ymetrics

Zlayers
[non_trainable_variables
\layer_metrics
,	variables
-regularization_losses
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
н
1trainable_variables
]layer_regularization_losses
^metrics

_layers
`non_trainable_variables
alayer_metrics
2	variables
3regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

b0
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ctotal
	dcount
e	variables
f	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

e	variables
{y
VARIABLE_VALUEAdam/Conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Conv2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Conv2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/Conv1_transpose/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/Conv1_transpose/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/Conv2_transpose/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/Conv2_transpose/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/output_layer/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_layer/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Conv2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Conv2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/Conv1_transpose/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/Conv1_transpose/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEAdam/Conv2_transpose/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUEAdam/Conv2_transpose/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/output_layer/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/output_layer/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
И
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Conv1/kernel
Conv1/biasConv2/kernel
Conv2/biasConv1_transpose/kernelConv1_transpose/biasConv2_transpose/kernelConv2_transpose/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_50764
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
н
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Conv1/kernel/Read/ReadVariableOpConv1/bias/Read/ReadVariableOp Conv2/kernel/Read/ReadVariableOpConv2/bias/Read/ReadVariableOp*Conv1_transpose/kernel/Read/ReadVariableOp(Conv1_transpose/bias/Read/ReadVariableOp*Conv2_transpose/kernel/Read/ReadVariableOp(Conv2_transpose/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/Conv1/kernel/m/Read/ReadVariableOp%Adam/Conv1/bias/m/Read/ReadVariableOp'Adam/Conv2/kernel/m/Read/ReadVariableOp%Adam/Conv2/bias/m/Read/ReadVariableOp1Adam/Conv1_transpose/kernel/m/Read/ReadVariableOp/Adam/Conv1_transpose/bias/m/Read/ReadVariableOp1Adam/Conv2_transpose/kernel/m/Read/ReadVariableOp/Adam/Conv2_transpose/bias/m/Read/ReadVariableOp.Adam/output_layer/kernel/m/Read/ReadVariableOp,Adam/output_layer/bias/m/Read/ReadVariableOp'Adam/Conv1/kernel/v/Read/ReadVariableOp%Adam/Conv1/bias/v/Read/ReadVariableOp'Adam/Conv2/kernel/v/Read/ReadVariableOp%Adam/Conv2/bias/v/Read/ReadVariableOp1Adam/Conv1_transpose/kernel/v/Read/ReadVariableOp/Adam/Conv1_transpose/bias/v/Read/ReadVariableOp1Adam/Conv2_transpose/kernel/v/Read/ReadVariableOp/Adam/Conv2_transpose/bias/v/Read/ReadVariableOp.Adam/output_layer/kernel/v/Read/ReadVariableOp,Adam/output_layer/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_51142
─
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1/kernel
Conv1/biasConv2/kernel
Conv2/biasConv1_transpose/kernelConv1_transpose/biasConv2_transpose/kernelConv2_transpose/biasoutput_layer/kerneloutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/Conv1/kernel/mAdam/Conv1/bias/mAdam/Conv2/kernel/mAdam/Conv2/bias/mAdam/Conv1_transpose/kernel/mAdam/Conv1_transpose/bias/mAdam/Conv2_transpose/kernel/mAdam/Conv2_transpose/bias/mAdam/output_layer/kernel/mAdam/output_layer/bias/mAdam/Conv1/kernel/vAdam/Conv1/bias/vAdam/Conv2/kernel/vAdam/Conv2/bias/vAdam/Conv1_transpose/kernel/vAdam/Conv1_transpose/bias/vAdam/Conv2_transpose/kernel/vAdam/Conv2_transpose/bias/vAdam/output_layer/kernel/vAdam/output_layer/bias/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_51263▌╦
Ъ
д
/__inference_Conv1_transpose_layer_call_fn_50394

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_503842
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ъ
д
/__inference_Conv2_transpose_layer_call_fn_50439

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_504292
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╛
Ъ
%__inference_Conv1_layer_call_fn_50968

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv1_layer_call_and_return_conditional_losses_504572
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┼`
∙
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50831

inputs>
$conv1_conv2d_readvariableop_resource: 3
%conv1_biasadd_readvariableop_resource: >
$conv2_conv2d_readvariableop_resource:  3
%conv2_biasadd_readvariableop_resource: R
8conv1_transpose_conv2d_transpose_readvariableop_resource:  =
/conv1_transpose_biasadd_readvariableop_resource: R
8conv2_transpose_conv2d_transpose_readvariableop_resource:  =
/conv2_transpose_biasadd_readvariableop_resource: E
+output_layer_conv2d_readvariableop_resource: :
,output_layer_biasadd_readvariableop_resource:
identityИвConv1/BiasAdd/ReadVariableOpвConv1/Conv2D/ReadVariableOpв&Conv1_transpose/BiasAdd/ReadVariableOpв/Conv1_transpose/conv2d_transpose/ReadVariableOpвConv2/BiasAdd/ReadVariableOpвConv2/Conv2D/ReadVariableOpв&Conv2_transpose/BiasAdd/ReadVariableOpв/Conv2_transpose/conv2d_transpose/ReadVariableOpв#output_layer/BiasAdd/ReadVariableOpв"output_layer/Conv2D/ReadVariableOpз
Conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv1/Conv2D/ReadVariableOp╡
Conv1/Conv2DConv2Dinputs#Conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv1/Conv2DЮ
Conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv1/BiasAdd/ReadVariableOpа
Conv1/BiasAddBiasAddConv1/Conv2D:output:0$Conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv1/BiasAddr

Conv1/ReluReluConv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2

Conv1/Reluп
Pool1/MaxPoolMaxPoolConv1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2
Pool1/MaxPoolз
Conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2/Conv2D/ReadVariableOp┼
Conv2/Conv2DConv2DPool1/MaxPool:output:0#Conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2/Conv2DЮ
Conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv2/BiasAdd/ReadVariableOpа
Conv2/BiasAddBiasAddConv2/Conv2D:output:0$Conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv2/BiasAddr

Conv2/ReluReluConv2/BiasAdd:output:0*
T0*/
_output_shapes
:          2

Conv2/Reluп
Pool2/MaxPoolMaxPoolConv2/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2
Pool2/MaxPoolt
Conv1_transpose/ShapeShapePool2/MaxPool:output:0*
T0*
_output_shapes
:2
Conv1_transpose/ShapeФ
#Conv1_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Conv1_transpose/strided_slice/stackШ
%Conv1_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv1_transpose/strided_slice/stack_1Ш
%Conv1_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv1_transpose/strided_slice/stack_2┬
Conv1_transpose/strided_sliceStridedSliceConv1_transpose/Shape:output:0,Conv1_transpose/strided_slice/stack:output:0.Conv1_transpose/strided_slice/stack_1:output:0.Conv1_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv1_transpose/strided_slicet
Conv1_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv1_transpose/stack/1t
Conv1_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Conv1_transpose/stack/2t
Conv1_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Conv1_transpose/stack/3Є
Conv1_transpose/stackPack&Conv1_transpose/strided_slice:output:0 Conv1_transpose/stack/1:output:0 Conv1_transpose/stack/2:output:0 Conv1_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv1_transpose/stackШ
%Conv1_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Conv1_transpose/strided_slice_1/stackЬ
'Conv1_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv1_transpose/strided_slice_1/stack_1Ь
'Conv1_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv1_transpose/strided_slice_1/stack_2╠
Conv1_transpose/strided_slice_1StridedSliceConv1_transpose/stack:output:0.Conv1_transpose/strided_slice_1/stack:output:00Conv1_transpose/strided_slice_1/stack_1:output:00Conv1_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Conv1_transpose/strided_slice_1у
/Conv1_transpose/conv2d_transpose/ReadVariableOpReadVariableOp8conv1_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype021
/Conv1_transpose/conv2d_transpose/ReadVariableOpо
 Conv1_transpose/conv2d_transposeConv2DBackpropInputConv1_transpose/stack:output:07Conv1_transpose/conv2d_transpose/ReadVariableOp:value:0Pool2/MaxPool:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2"
 Conv1_transpose/conv2d_transpose╝
&Conv1_transpose/BiasAdd/ReadVariableOpReadVariableOp/conv1_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&Conv1_transpose/BiasAdd/ReadVariableOp╥
Conv1_transpose/BiasAddBiasAdd)Conv1_transpose/conv2d_transpose:output:0.Conv1_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv1_transpose/BiasAddР
Conv1_transpose/ReluRelu Conv1_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:          2
Conv1_transpose/ReluА
Conv2_transpose/ShapeShape"Conv1_transpose/Relu:activations:0*
T0*
_output_shapes
:2
Conv2_transpose/ShapeФ
#Conv2_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Conv2_transpose/strided_slice/stackШ
%Conv2_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv2_transpose/strided_slice/stack_1Ш
%Conv2_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv2_transpose/strided_slice/stack_2┬
Conv2_transpose/strided_sliceStridedSliceConv2_transpose/Shape:output:0,Conv2_transpose/strided_slice/stack:output:0.Conv2_transpose/strided_slice/stack_1:output:0.Conv2_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv2_transpose/strided_slicet
Conv2_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv2_transpose/stack/1t
Conv2_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Conv2_transpose/stack/2t
Conv2_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Conv2_transpose/stack/3Є
Conv2_transpose/stackPack&Conv2_transpose/strided_slice:output:0 Conv2_transpose/stack/1:output:0 Conv2_transpose/stack/2:output:0 Conv2_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv2_transpose/stackШ
%Conv2_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Conv2_transpose/strided_slice_1/stackЬ
'Conv2_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2_transpose/strided_slice_1/stack_1Ь
'Conv2_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2_transpose/strided_slice_1/stack_2╠
Conv2_transpose/strided_slice_1StridedSliceConv2_transpose/stack:output:0.Conv2_transpose/strided_slice_1/stack:output:00Conv2_transpose/strided_slice_1/stack_1:output:00Conv2_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Conv2_transpose/strided_slice_1у
/Conv2_transpose/conv2d_transpose/ReadVariableOpReadVariableOp8conv2_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype021
/Conv2_transpose/conv2d_transpose/ReadVariableOp║
 Conv2_transpose/conv2d_transposeConv2DBackpropInputConv2_transpose/stack:output:07Conv2_transpose/conv2d_transpose/ReadVariableOp:value:0"Conv1_transpose/Relu:activations:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2"
 Conv2_transpose/conv2d_transpose╝
&Conv2_transpose/BiasAdd/ReadVariableOpReadVariableOp/conv2_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&Conv2_transpose/BiasAdd/ReadVariableOp╥
Conv2_transpose/BiasAddBiasAdd)Conv2_transpose/conv2d_transpose:output:0.Conv2_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv2_transpose/BiasAddР
Conv2_transpose/ReluRelu Conv2_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:          2
Conv2_transpose/Relu╝
"output_layer/Conv2D/ReadVariableOpReadVariableOp+output_layer_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"output_layer/Conv2D/ReadVariableOpц
output_layer/Conv2DConv2D"Conv2_transpose/Relu:activations:0*output_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
output_layer/Conv2D│
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOp╝
output_layer/BiasAddBiasAddoutput_layer/Conv2D:output:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
output_layer/BiasAddР
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*/
_output_shapes
:         2
output_layer/Sigmoidя
IdentityIdentityoutput_layer/Sigmoid:y:0^Conv1/BiasAdd/ReadVariableOp^Conv1/Conv2D/ReadVariableOp'^Conv1_transpose/BiasAdd/ReadVariableOp0^Conv1_transpose/conv2d_transpose/ReadVariableOp^Conv2/BiasAdd/ReadVariableOp^Conv2/Conv2D/ReadVariableOp'^Conv2_transpose/BiasAdd/ReadVariableOp0^Conv2_transpose/conv2d_transpose/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2<
Conv1/BiasAdd/ReadVariableOpConv1/BiasAdd/ReadVariableOp2:
Conv1/Conv2D/ReadVariableOpConv1/Conv2D/ReadVariableOp2P
&Conv1_transpose/BiasAdd/ReadVariableOp&Conv1_transpose/BiasAdd/ReadVariableOp2b
/Conv1_transpose/conv2d_transpose/ReadVariableOp/Conv1_transpose/conv2d_transpose/ReadVariableOp2<
Conv2/BiasAdd/ReadVariableOpConv2/BiasAdd/ReadVariableOp2:
Conv2/Conv2D/ReadVariableOpConv2/Conv2D/ReadVariableOp2P
&Conv2_transpose/BiasAdd/ReadVariableOp&Conv2_transpose/BiasAdd/ReadVariableOp2b
/Conv2_transpose/conv2d_transpose/ReadVariableOp/Conv2_transpose/conv2d_transpose/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/Conv2D/ReadVariableOp"output_layer/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ъ

Ю
1__inference_AutoEncoder-Model_layer_call_fn_50923

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_505102
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ф
б
,__inference_output_layer_layer_call_fn_51008

inputs!
unknown: 
	unknown_0:
identityИвStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_505032
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
▒
╢
 __inference__wrapped_model_50325
input_1P
6autoencoder_model_conv1_conv2d_readvariableop_resource: E
7autoencoder_model_conv1_biasadd_readvariableop_resource: P
6autoencoder_model_conv2_conv2d_readvariableop_resource:  E
7autoencoder_model_conv2_biasadd_readvariableop_resource: d
Jautoencoder_model_conv1_transpose_conv2d_transpose_readvariableop_resource:  O
Aautoencoder_model_conv1_transpose_biasadd_readvariableop_resource: d
Jautoencoder_model_conv2_transpose_conv2d_transpose_readvariableop_resource:  O
Aautoencoder_model_conv2_transpose_biasadd_readvariableop_resource: W
=autoencoder_model_output_layer_conv2d_readvariableop_resource: L
>autoencoder_model_output_layer_biasadd_readvariableop_resource:
identityИв.AutoEncoder-Model/Conv1/BiasAdd/ReadVariableOpв-AutoEncoder-Model/Conv1/Conv2D/ReadVariableOpв8AutoEncoder-Model/Conv1_transpose/BiasAdd/ReadVariableOpвAAutoEncoder-Model/Conv1_transpose/conv2d_transpose/ReadVariableOpв.AutoEncoder-Model/Conv2/BiasAdd/ReadVariableOpв-AutoEncoder-Model/Conv2/Conv2D/ReadVariableOpв8AutoEncoder-Model/Conv2_transpose/BiasAdd/ReadVariableOpвAAutoEncoder-Model/Conv2_transpose/conv2d_transpose/ReadVariableOpв5AutoEncoder-Model/output_layer/BiasAdd/ReadVariableOpв4AutoEncoder-Model/output_layer/Conv2D/ReadVariableOp▌
-AutoEncoder-Model/Conv1/Conv2D/ReadVariableOpReadVariableOp6autoencoder_model_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-AutoEncoder-Model/Conv1/Conv2D/ReadVariableOpь
AutoEncoder-Model/Conv1/Conv2DConv2Dinput_15AutoEncoder-Model/Conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2 
AutoEncoder-Model/Conv1/Conv2D╘
.AutoEncoder-Model/Conv1/BiasAdd/ReadVariableOpReadVariableOp7autoencoder_model_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.AutoEncoder-Model/Conv1/BiasAdd/ReadVariableOpш
AutoEncoder-Model/Conv1/BiasAddBiasAdd'AutoEncoder-Model/Conv1/Conv2D:output:06AutoEncoder-Model/Conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2!
AutoEncoder-Model/Conv1/BiasAddи
AutoEncoder-Model/Conv1/ReluRelu(AutoEncoder-Model/Conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
AutoEncoder-Model/Conv1/Reluх
AutoEncoder-Model/Pool1/MaxPoolMaxPool*AutoEncoder-Model/Conv1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2!
AutoEncoder-Model/Pool1/MaxPool▌
-AutoEncoder-Model/Conv2/Conv2D/ReadVariableOpReadVariableOp6autoencoder_model_conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-AutoEncoder-Model/Conv2/Conv2D/ReadVariableOpН
AutoEncoder-Model/Conv2/Conv2DConv2D(AutoEncoder-Model/Pool1/MaxPool:output:05AutoEncoder-Model/Conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2 
AutoEncoder-Model/Conv2/Conv2D╘
.AutoEncoder-Model/Conv2/BiasAdd/ReadVariableOpReadVariableOp7autoencoder_model_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.AutoEncoder-Model/Conv2/BiasAdd/ReadVariableOpш
AutoEncoder-Model/Conv2/BiasAddBiasAdd'AutoEncoder-Model/Conv2/Conv2D:output:06AutoEncoder-Model/Conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2!
AutoEncoder-Model/Conv2/BiasAddи
AutoEncoder-Model/Conv2/ReluRelu(AutoEncoder-Model/Conv2/BiasAdd:output:0*
T0*/
_output_shapes
:          2
AutoEncoder-Model/Conv2/Reluх
AutoEncoder-Model/Pool2/MaxPoolMaxPool*AutoEncoder-Model/Conv2/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2!
AutoEncoder-Model/Pool2/MaxPoolк
'AutoEncoder-Model/Conv1_transpose/ShapeShape(AutoEncoder-Model/Pool2/MaxPool:output:0*
T0*
_output_shapes
:2)
'AutoEncoder-Model/Conv1_transpose/Shape╕
5AutoEncoder-Model/Conv1_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5AutoEncoder-Model/Conv1_transpose/strided_slice/stack╝
7AutoEncoder-Model/Conv1_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7AutoEncoder-Model/Conv1_transpose/strided_slice/stack_1╝
7AutoEncoder-Model/Conv1_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7AutoEncoder-Model/Conv1_transpose/strided_slice/stack_2о
/AutoEncoder-Model/Conv1_transpose/strided_sliceStridedSlice0AutoEncoder-Model/Conv1_transpose/Shape:output:0>AutoEncoder-Model/Conv1_transpose/strided_slice/stack:output:0@AutoEncoder-Model/Conv1_transpose/strided_slice/stack_1:output:0@AutoEncoder-Model/Conv1_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/AutoEncoder-Model/Conv1_transpose/strided_sliceШ
)AutoEncoder-Model/Conv1_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)AutoEncoder-Model/Conv1_transpose/stack/1Ш
)AutoEncoder-Model/Conv1_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)AutoEncoder-Model/Conv1_transpose/stack/2Ш
)AutoEncoder-Model/Conv1_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2+
)AutoEncoder-Model/Conv1_transpose/stack/3▐
'AutoEncoder-Model/Conv1_transpose/stackPack8AutoEncoder-Model/Conv1_transpose/strided_slice:output:02AutoEncoder-Model/Conv1_transpose/stack/1:output:02AutoEncoder-Model/Conv1_transpose/stack/2:output:02AutoEncoder-Model/Conv1_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'AutoEncoder-Model/Conv1_transpose/stack╝
7AutoEncoder-Model/Conv1_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7AutoEncoder-Model/Conv1_transpose/strided_slice_1/stack└
9AutoEncoder-Model/Conv1_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9AutoEncoder-Model/Conv1_transpose/strided_slice_1/stack_1└
9AutoEncoder-Model/Conv1_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9AutoEncoder-Model/Conv1_transpose/strided_slice_1/stack_2╕
1AutoEncoder-Model/Conv1_transpose/strided_slice_1StridedSlice0AutoEncoder-Model/Conv1_transpose/stack:output:0@AutoEncoder-Model/Conv1_transpose/strided_slice_1/stack:output:0BAutoEncoder-Model/Conv1_transpose/strided_slice_1/stack_1:output:0BAutoEncoder-Model/Conv1_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1AutoEncoder-Model/Conv1_transpose/strided_slice_1Щ
AAutoEncoder-Model/Conv1_transpose/conv2d_transpose/ReadVariableOpReadVariableOpJautoencoder_model_conv1_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02C
AAutoEncoder-Model/Conv1_transpose/conv2d_transpose/ReadVariableOpИ
2AutoEncoder-Model/Conv1_transpose/conv2d_transposeConv2DBackpropInput0AutoEncoder-Model/Conv1_transpose/stack:output:0IAutoEncoder-Model/Conv1_transpose/conv2d_transpose/ReadVariableOp:value:0(AutoEncoder-Model/Pool2/MaxPool:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
24
2AutoEncoder-Model/Conv1_transpose/conv2d_transposeЄ
8AutoEncoder-Model/Conv1_transpose/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_model_conv1_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8AutoEncoder-Model/Conv1_transpose/BiasAdd/ReadVariableOpЪ
)AutoEncoder-Model/Conv1_transpose/BiasAddBiasAdd;AutoEncoder-Model/Conv1_transpose/conv2d_transpose:output:0@AutoEncoder-Model/Conv1_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2+
)AutoEncoder-Model/Conv1_transpose/BiasAdd╞
&AutoEncoder-Model/Conv1_transpose/ReluRelu2AutoEncoder-Model/Conv1_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:          2(
&AutoEncoder-Model/Conv1_transpose/Relu╢
'AutoEncoder-Model/Conv2_transpose/ShapeShape4AutoEncoder-Model/Conv1_transpose/Relu:activations:0*
T0*
_output_shapes
:2)
'AutoEncoder-Model/Conv2_transpose/Shape╕
5AutoEncoder-Model/Conv2_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5AutoEncoder-Model/Conv2_transpose/strided_slice/stack╝
7AutoEncoder-Model/Conv2_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7AutoEncoder-Model/Conv2_transpose/strided_slice/stack_1╝
7AutoEncoder-Model/Conv2_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7AutoEncoder-Model/Conv2_transpose/strided_slice/stack_2о
/AutoEncoder-Model/Conv2_transpose/strided_sliceStridedSlice0AutoEncoder-Model/Conv2_transpose/Shape:output:0>AutoEncoder-Model/Conv2_transpose/strided_slice/stack:output:0@AutoEncoder-Model/Conv2_transpose/strided_slice/stack_1:output:0@AutoEncoder-Model/Conv2_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/AutoEncoder-Model/Conv2_transpose/strided_sliceШ
)AutoEncoder-Model/Conv2_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)AutoEncoder-Model/Conv2_transpose/stack/1Ш
)AutoEncoder-Model/Conv2_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)AutoEncoder-Model/Conv2_transpose/stack/2Ш
)AutoEncoder-Model/Conv2_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2+
)AutoEncoder-Model/Conv2_transpose/stack/3▐
'AutoEncoder-Model/Conv2_transpose/stackPack8AutoEncoder-Model/Conv2_transpose/strided_slice:output:02AutoEncoder-Model/Conv2_transpose/stack/1:output:02AutoEncoder-Model/Conv2_transpose/stack/2:output:02AutoEncoder-Model/Conv2_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2)
'AutoEncoder-Model/Conv2_transpose/stack╝
7AutoEncoder-Model/Conv2_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7AutoEncoder-Model/Conv2_transpose/strided_slice_1/stack└
9AutoEncoder-Model/Conv2_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9AutoEncoder-Model/Conv2_transpose/strided_slice_1/stack_1└
9AutoEncoder-Model/Conv2_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9AutoEncoder-Model/Conv2_transpose/strided_slice_1/stack_2╕
1AutoEncoder-Model/Conv2_transpose/strided_slice_1StridedSlice0AutoEncoder-Model/Conv2_transpose/stack:output:0@AutoEncoder-Model/Conv2_transpose/strided_slice_1/stack:output:0BAutoEncoder-Model/Conv2_transpose/strided_slice_1/stack_1:output:0BAutoEncoder-Model/Conv2_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1AutoEncoder-Model/Conv2_transpose/strided_slice_1Щ
AAutoEncoder-Model/Conv2_transpose/conv2d_transpose/ReadVariableOpReadVariableOpJautoencoder_model_conv2_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02C
AAutoEncoder-Model/Conv2_transpose/conv2d_transpose/ReadVariableOpФ
2AutoEncoder-Model/Conv2_transpose/conv2d_transposeConv2DBackpropInput0AutoEncoder-Model/Conv2_transpose/stack:output:0IAutoEncoder-Model/Conv2_transpose/conv2d_transpose/ReadVariableOp:value:04AutoEncoder-Model/Conv1_transpose/Relu:activations:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
24
2AutoEncoder-Model/Conv2_transpose/conv2d_transposeЄ
8AutoEncoder-Model/Conv2_transpose/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_model_conv2_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8AutoEncoder-Model/Conv2_transpose/BiasAdd/ReadVariableOpЪ
)AutoEncoder-Model/Conv2_transpose/BiasAddBiasAdd;AutoEncoder-Model/Conv2_transpose/conv2d_transpose:output:0@AutoEncoder-Model/Conv2_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2+
)AutoEncoder-Model/Conv2_transpose/BiasAdd╞
&AutoEncoder-Model/Conv2_transpose/ReluRelu2AutoEncoder-Model/Conv2_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:          2(
&AutoEncoder-Model/Conv2_transpose/ReluЄ
4AutoEncoder-Model/output_layer/Conv2D/ReadVariableOpReadVariableOp=autoencoder_model_output_layer_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype026
4AutoEncoder-Model/output_layer/Conv2D/ReadVariableOpо
%AutoEncoder-Model/output_layer/Conv2DConv2D4AutoEncoder-Model/Conv2_transpose/Relu:activations:0<AutoEncoder-Model/output_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2'
%AutoEncoder-Model/output_layer/Conv2Dщ
5AutoEncoder-Model/output_layer/BiasAdd/ReadVariableOpReadVariableOp>autoencoder_model_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5AutoEncoder-Model/output_layer/BiasAdd/ReadVariableOpД
&AutoEncoder-Model/output_layer/BiasAddBiasAdd.AutoEncoder-Model/output_layer/Conv2D:output:0=AutoEncoder-Model/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2(
&AutoEncoder-Model/output_layer/BiasAdd╞
&AutoEncoder-Model/output_layer/SigmoidSigmoid/AutoEncoder-Model/output_layer/BiasAdd:output:0*
T0*/
_output_shapes
:         2(
&AutoEncoder-Model/output_layer/Sigmoid╡
IdentityIdentity*AutoEncoder-Model/output_layer/Sigmoid:y:0/^AutoEncoder-Model/Conv1/BiasAdd/ReadVariableOp.^AutoEncoder-Model/Conv1/Conv2D/ReadVariableOp9^AutoEncoder-Model/Conv1_transpose/BiasAdd/ReadVariableOpB^AutoEncoder-Model/Conv1_transpose/conv2d_transpose/ReadVariableOp/^AutoEncoder-Model/Conv2/BiasAdd/ReadVariableOp.^AutoEncoder-Model/Conv2/Conv2D/ReadVariableOp9^AutoEncoder-Model/Conv2_transpose/BiasAdd/ReadVariableOpB^AutoEncoder-Model/Conv2_transpose/conv2d_transpose/ReadVariableOp6^AutoEncoder-Model/output_layer/BiasAdd/ReadVariableOp5^AutoEncoder-Model/output_layer/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2`
.AutoEncoder-Model/Conv1/BiasAdd/ReadVariableOp.AutoEncoder-Model/Conv1/BiasAdd/ReadVariableOp2^
-AutoEncoder-Model/Conv1/Conv2D/ReadVariableOp-AutoEncoder-Model/Conv1/Conv2D/ReadVariableOp2t
8AutoEncoder-Model/Conv1_transpose/BiasAdd/ReadVariableOp8AutoEncoder-Model/Conv1_transpose/BiasAdd/ReadVariableOp2Ж
AAutoEncoder-Model/Conv1_transpose/conv2d_transpose/ReadVariableOpAAutoEncoder-Model/Conv1_transpose/conv2d_transpose/ReadVariableOp2`
.AutoEncoder-Model/Conv2/BiasAdd/ReadVariableOp.AutoEncoder-Model/Conv2/BiasAdd/ReadVariableOp2^
-AutoEncoder-Model/Conv2/Conv2D/ReadVariableOp-AutoEncoder-Model/Conv2/Conv2D/ReadVariableOp2t
8AutoEncoder-Model/Conv2_transpose/BiasAdd/ReadVariableOp8AutoEncoder-Model/Conv2_transpose/BiasAdd/ReadVariableOp2Ж
AAutoEncoder-Model/Conv2_transpose/conv2d_transpose/ReadVariableOpAAutoEncoder-Model/Conv2_transpose/conv2d_transpose/ReadVariableOp2n
5AutoEncoder-Model/output_layer/BiasAdd/ReadVariableOp5AutoEncoder-Model/output_layer/BiasAdd/ReadVariableOp2l
4AutoEncoder-Model/output_layer/Conv2D/ReadVariableOp4AutoEncoder-Model/output_layer/Conv2D/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
еR
э
__inference__traced_save_51142
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop5
1savev2_conv1_transpose_kernel_read_readvariableop3
/savev2_conv1_transpose_bias_read_readvariableop5
1savev2_conv2_transpose_kernel_read_readvariableop3
/savev2_conv2_transpose_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableop<
8savev2_adam_conv1_transpose_kernel_m_read_readvariableop:
6savev2_adam_conv1_transpose_bias_m_read_readvariableop<
8savev2_adam_conv2_transpose_kernel_m_read_readvariableop:
6savev2_adam_conv2_transpose_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableop<
8savev2_adam_conv1_transpose_kernel_v_read_readvariableop:
6savev2_adam_conv1_transpose_bias_v_read_readvariableop<
8savev2_adam_conv2_transpose_kernel_v_read_readvariableop:
6savev2_adam_conv2_transpose_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameФ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ж
valueЬBЩ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices═
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop1savev2_conv1_transpose_kernel_read_readvariableop/savev2_conv1_transpose_bias_read_readvariableop1savev2_conv2_transpose_kernel_read_readvariableop/savev2_conv2_transpose_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop8savev2_adam_conv1_transpose_kernel_m_read_readvariableop6savev2_adam_conv1_transpose_bias_m_read_readvariableop8savev2_adam_conv2_transpose_kernel_m_read_readvariableop6savev2_adam_conv2_transpose_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop8savev2_adam_conv1_transpose_kernel_v_read_readvariableop6savev2_adam_conv1_transpose_bias_v_read_readvariableop8savev2_adam_conv2_transpose_kernel_v_read_readvariableop6savev2_adam_conv2_transpose_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*П
_input_shapes¤
·: : : :  : :  : :  : : :: : : : : : : : : :  : :  : :  : : :: : :  : :  : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: : 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :, (
&
_output_shapes
:  : !

_output_shapes
: :,"(
&
_output_shapes
:  : #

_output_shapes
: :,$(
&
_output_shapes
: : %

_output_shapes
::&

_output_shapes
: 
ъ

Ю
1__inference_AutoEncoder-Model_layer_call_fn_50948

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_506212
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Е
А
G__inference_output_layer_layer_call_and_return_conditional_losses_50503

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2	
Sigmoidк
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Е
А
G__inference_output_layer_layer_call_and_return_conditional_losses_50999

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2	
Sigmoidк
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
─
A
%__inference_Pool1_layer_call_fn_50337

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool1_layer_call_and_return_conditional_losses_503312
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
э

Я
1__inference_AutoEncoder-Model_layer_call_fn_50533
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_505102
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
╛
Ъ
%__inference_Conv2_layer_call_fn_50988

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv2_layer_call_and_return_conditional_losses_504752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
▀$
ж
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50621

inputs%
conv1_50593: 
conv1_50595: %
conv2_50599:  
conv2_50601: /
conv1_transpose_50605:  #
conv1_transpose_50607: /
conv2_transpose_50610:  #
conv2_transpose_50612: ,
output_layer_50615:  
output_layer_50617:
identityИвConv1/StatefulPartitionedCallв'Conv1_transpose/StatefulPartitionedCallвConv2/StatefulPartitionedCallв'Conv2_transpose/StatefulPartitionedCallв$output_layer/StatefulPartitionedCallН
Conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_50593conv1_50595*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv1_layer_call_and_return_conditional_losses_504572
Conv1/StatefulPartitionedCallї
Pool1/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool1_layer_call_and_return_conditional_losses_503312
Pool1/PartitionedCallе
Conv2/StatefulPartitionedCallStatefulPartitionedCallPool1/PartitionedCall:output:0conv2_50599conv2_50601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv2_layer_call_and_return_conditional_losses_504752
Conv2/StatefulPartitionedCallї
Pool2/PartitionedCallPartitionedCall&Conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool2_layer_call_and_return_conditional_losses_503432
Pool2/PartitionedCallщ
'Conv1_transpose/StatefulPartitionedCallStatefulPartitionedCallPool2/PartitionedCall:output:0conv1_transpose_50605conv1_transpose_50607*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_503842)
'Conv1_transpose/StatefulPartitionedCall√
'Conv2_transpose/StatefulPartitionedCallStatefulPartitionedCall0Conv1_transpose/StatefulPartitionedCall:output:0conv2_transpose_50610conv2_transpose_50612*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_504292)
'Conv2_transpose/StatefulPartitionedCallь
$output_layer/StatefulPartitionedCallStatefulPartitionedCall0Conv2_transpose/StatefulPartitionedCall:output:0output_layer_50615output_layer_50617*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_505032&
$output_layer/StatefulPartitionedCall╓
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall(^Conv1_transpose/StatefulPartitionedCall^Conv2/StatefulPartitionedCall(^Conv2_transpose/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2R
'Conv1_transpose/StatefulPartitionedCall'Conv1_transpose/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2R
'Conv2_transpose/StatefulPartitionedCall'Conv2_transpose/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
т$
з
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50731
input_1%
conv1_50703: 
conv1_50705: %
conv2_50709:  
conv2_50711: /
conv1_transpose_50715:  #
conv1_transpose_50717: /
conv2_transpose_50720:  #
conv2_transpose_50722: ,
output_layer_50725:  
output_layer_50727:
identityИвConv1/StatefulPartitionedCallв'Conv1_transpose/StatefulPartitionedCallвConv2/StatefulPartitionedCallв'Conv2_transpose/StatefulPartitionedCallв$output_layer/StatefulPartitionedCallО
Conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_50703conv1_50705*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv1_layer_call_and_return_conditional_losses_504572
Conv1/StatefulPartitionedCallї
Pool1/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool1_layer_call_and_return_conditional_losses_503312
Pool1/PartitionedCallе
Conv2/StatefulPartitionedCallStatefulPartitionedCallPool1/PartitionedCall:output:0conv2_50709conv2_50711*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv2_layer_call_and_return_conditional_losses_504752
Conv2/StatefulPartitionedCallї
Pool2/PartitionedCallPartitionedCall&Conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool2_layer_call_and_return_conditional_losses_503432
Pool2/PartitionedCallщ
'Conv1_transpose/StatefulPartitionedCallStatefulPartitionedCallPool2/PartitionedCall:output:0conv1_transpose_50715conv1_transpose_50717*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_503842)
'Conv1_transpose/StatefulPartitionedCall√
'Conv2_transpose/StatefulPartitionedCallStatefulPartitionedCall0Conv1_transpose/StatefulPartitionedCall:output:0conv2_transpose_50720conv2_transpose_50722*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_504292)
'Conv2_transpose/StatefulPartitionedCallь
$output_layer/StatefulPartitionedCallStatefulPartitionedCall0Conv2_transpose/StatefulPartitionedCall:output:0output_layer_50725output_layer_50727*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_505032&
$output_layer/StatefulPartitionedCall╓
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall(^Conv1_transpose/StatefulPartitionedCall^Conv2/StatefulPartitionedCall(^Conv2_transpose/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2R
'Conv1_transpose/StatefulPartitionedCall'Conv1_transpose/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2R
'Conv2_transpose/StatefulPartitionedCall'Conv2_transpose/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
д%
Ч
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_50384

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_transpose/ReadVariableOpЁ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu╗
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
д%
Ч
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_50429

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3В
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_transpose/ReadVariableOpЁ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpд
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu╗
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Я
\
@__inference_Pool1_layer_call_and_return_conditional_losses_50331

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▀$
ж
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50510

inputs%
conv1_50458: 
conv1_50460: %
conv2_50476:  
conv2_50478: /
conv1_transpose_50482:  #
conv1_transpose_50484: /
conv2_transpose_50487:  #
conv2_transpose_50489: ,
output_layer_50504:  
output_layer_50506:
identityИвConv1/StatefulPartitionedCallв'Conv1_transpose/StatefulPartitionedCallвConv2/StatefulPartitionedCallв'Conv2_transpose/StatefulPartitionedCallв$output_layer/StatefulPartitionedCallН
Conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_50458conv1_50460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv1_layer_call_and_return_conditional_losses_504572
Conv1/StatefulPartitionedCallї
Pool1/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool1_layer_call_and_return_conditional_losses_503312
Pool1/PartitionedCallе
Conv2/StatefulPartitionedCallStatefulPartitionedCallPool1/PartitionedCall:output:0conv2_50476conv2_50478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv2_layer_call_and_return_conditional_losses_504752
Conv2/StatefulPartitionedCallї
Pool2/PartitionedCallPartitionedCall&Conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool2_layer_call_and_return_conditional_losses_503432
Pool2/PartitionedCallщ
'Conv1_transpose/StatefulPartitionedCallStatefulPartitionedCallPool2/PartitionedCall:output:0conv1_transpose_50482conv1_transpose_50484*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_503842)
'Conv1_transpose/StatefulPartitionedCall√
'Conv2_transpose/StatefulPartitionedCallStatefulPartitionedCall0Conv1_transpose/StatefulPartitionedCall:output:0conv2_transpose_50487conv2_transpose_50489*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_504292)
'Conv2_transpose/StatefulPartitionedCallь
$output_layer/StatefulPartitionedCallStatefulPartitionedCall0Conv2_transpose/StatefulPartitionedCall:output:0output_layer_50504output_layer_50506*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_505032&
$output_layer/StatefulPartitionedCall╓
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall(^Conv1_transpose/StatefulPartitionedCall^Conv2/StatefulPartitionedCall(^Conv2_transpose/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2R
'Conv1_transpose/StatefulPartitionedCall'Conv1_transpose/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2R
'Conv2_transpose/StatefulPartitionedCall'Conv2_transpose/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
э

Я
1__inference_AutoEncoder-Model_layer_call_fn_50669
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_506212
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
─б
ч
!__inference__traced_restore_51263
file_prefix7
assignvariableop_conv1_kernel: +
assignvariableop_1_conv1_bias: 9
assignvariableop_2_conv2_kernel:  +
assignvariableop_3_conv2_bias: C
)assignvariableop_4_conv1_transpose_kernel:  5
'assignvariableop_5_conv1_transpose_bias: C
)assignvariableop_6_conv2_transpose_kernel:  5
'assignvariableop_7_conv2_transpose_bias: @
&assignvariableop_8_output_layer_kernel: 2
$assignvariableop_9_output_layer_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: A
'assignvariableop_17_adam_conv1_kernel_m: 3
%assignvariableop_18_adam_conv1_bias_m: A
'assignvariableop_19_adam_conv2_kernel_m:  3
%assignvariableop_20_adam_conv2_bias_m: K
1assignvariableop_21_adam_conv1_transpose_kernel_m:  =
/assignvariableop_22_adam_conv1_transpose_bias_m: K
1assignvariableop_23_adam_conv2_transpose_kernel_m:  =
/assignvariableop_24_adam_conv2_transpose_bias_m: H
.assignvariableop_25_adam_output_layer_kernel_m: :
,assignvariableop_26_adam_output_layer_bias_m:A
'assignvariableop_27_adam_conv1_kernel_v: 3
%assignvariableop_28_adam_conv1_bias_v: A
'assignvariableop_29_adam_conv2_kernel_v:  3
%assignvariableop_30_adam_conv2_bias_v: K
1assignvariableop_31_adam_conv1_transpose_kernel_v:  =
/assignvariableop_32_adam_conv1_transpose_bias_v: K
1assignvariableop_33_adam_conv2_transpose_kernel_v:  =
/assignvariableop_34_adam_conv2_transpose_bias_v: H
.assignvariableop_35_adam_output_layer_kernel_v: :
,assignvariableop_36_adam_output_layer_bias_v:
identity_38ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Ъ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ж
valueЬBЩ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names┌
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*о
_output_shapesЫ
Ш::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1в
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2д
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3в
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4о
AssignVariableOp_4AssignVariableOp)assignvariableop_4_conv1_transpose_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5м
AssignVariableOp_5AssignVariableOp'assignvariableop_5_conv1_transpose_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6о
AssignVariableOp_6AssignVariableOp)assignvariableop_6_conv2_transpose_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7м
AssignVariableOp_7AssignVariableOp'assignvariableop_7_conv2_transpose_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8л
AssignVariableOp_8AssignVariableOp&assignvariableop_8_output_layer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9й
AssignVariableOp_9AssignVariableOp$assignvariableop_9_output_layer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10е
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11з
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12з
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ж
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14о
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15б
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16б
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17п
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_conv1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18н
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_conv1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19п
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_conv2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20н
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_conv2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╣
AssignVariableOp_21AssignVariableOp1assignvariableop_21_adam_conv1_transpose_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╖
AssignVariableOp_22AssignVariableOp/assignvariableop_22_adam_conv1_transpose_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╣
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_conv2_transpose_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╖
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_conv2_transpose_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╢
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_output_layer_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26┤
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_output_layer_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27п
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_conv1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28н
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_conv1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29п
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_conv2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30н
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_conv2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╣
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_conv1_transpose_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╖
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_conv1_transpose_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╣
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_conv2_transpose_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╖
AssignVariableOp_34AssignVariableOp/assignvariableop_34_adam_conv2_transpose_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╢
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_output_layer_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36┤
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_output_layer_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpМ
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37 
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┼`
∙
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50898

inputs>
$conv1_conv2d_readvariableop_resource: 3
%conv1_biasadd_readvariableop_resource: >
$conv2_conv2d_readvariableop_resource:  3
%conv2_biasadd_readvariableop_resource: R
8conv1_transpose_conv2d_transpose_readvariableop_resource:  =
/conv1_transpose_biasadd_readvariableop_resource: R
8conv2_transpose_conv2d_transpose_readvariableop_resource:  =
/conv2_transpose_biasadd_readvariableop_resource: E
+output_layer_conv2d_readvariableop_resource: :
,output_layer_biasadd_readvariableop_resource:
identityИвConv1/BiasAdd/ReadVariableOpвConv1/Conv2D/ReadVariableOpв&Conv1_transpose/BiasAdd/ReadVariableOpв/Conv1_transpose/conv2d_transpose/ReadVariableOpвConv2/BiasAdd/ReadVariableOpвConv2/Conv2D/ReadVariableOpв&Conv2_transpose/BiasAdd/ReadVariableOpв/Conv2_transpose/conv2d_transpose/ReadVariableOpв#output_layer/BiasAdd/ReadVariableOpв"output_layer/Conv2D/ReadVariableOpз
Conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv1/Conv2D/ReadVariableOp╡
Conv1/Conv2DConv2Dinputs#Conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv1/Conv2DЮ
Conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv1/BiasAdd/ReadVariableOpа
Conv1/BiasAddBiasAddConv1/Conv2D:output:0$Conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv1/BiasAddr

Conv1/ReluReluConv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2

Conv1/Reluп
Pool1/MaxPoolMaxPoolConv1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2
Pool1/MaxPoolз
Conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2/Conv2D/ReadVariableOp┼
Conv2/Conv2DConv2DPool1/MaxPool:output:0#Conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2/Conv2DЮ
Conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv2/BiasAdd/ReadVariableOpа
Conv2/BiasAddBiasAddConv2/Conv2D:output:0$Conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv2/BiasAddr

Conv2/ReluReluConv2/BiasAdd:output:0*
T0*/
_output_shapes
:          2

Conv2/Reluп
Pool2/MaxPoolMaxPoolConv2/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2
Pool2/MaxPoolt
Conv1_transpose/ShapeShapePool2/MaxPool:output:0*
T0*
_output_shapes
:2
Conv1_transpose/ShapeФ
#Conv1_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Conv1_transpose/strided_slice/stackШ
%Conv1_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv1_transpose/strided_slice/stack_1Ш
%Conv1_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv1_transpose/strided_slice/stack_2┬
Conv1_transpose/strided_sliceStridedSliceConv1_transpose/Shape:output:0,Conv1_transpose/strided_slice/stack:output:0.Conv1_transpose/strided_slice/stack_1:output:0.Conv1_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv1_transpose/strided_slicet
Conv1_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv1_transpose/stack/1t
Conv1_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Conv1_transpose/stack/2t
Conv1_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Conv1_transpose/stack/3Є
Conv1_transpose/stackPack&Conv1_transpose/strided_slice:output:0 Conv1_transpose/stack/1:output:0 Conv1_transpose/stack/2:output:0 Conv1_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv1_transpose/stackШ
%Conv1_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Conv1_transpose/strided_slice_1/stackЬ
'Conv1_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv1_transpose/strided_slice_1/stack_1Ь
'Conv1_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv1_transpose/strided_slice_1/stack_2╠
Conv1_transpose/strided_slice_1StridedSliceConv1_transpose/stack:output:0.Conv1_transpose/strided_slice_1/stack:output:00Conv1_transpose/strided_slice_1/stack_1:output:00Conv1_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Conv1_transpose/strided_slice_1у
/Conv1_transpose/conv2d_transpose/ReadVariableOpReadVariableOp8conv1_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype021
/Conv1_transpose/conv2d_transpose/ReadVariableOpо
 Conv1_transpose/conv2d_transposeConv2DBackpropInputConv1_transpose/stack:output:07Conv1_transpose/conv2d_transpose/ReadVariableOp:value:0Pool2/MaxPool:output:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2"
 Conv1_transpose/conv2d_transpose╝
&Conv1_transpose/BiasAdd/ReadVariableOpReadVariableOp/conv1_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&Conv1_transpose/BiasAdd/ReadVariableOp╥
Conv1_transpose/BiasAddBiasAdd)Conv1_transpose/conv2d_transpose:output:0.Conv1_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv1_transpose/BiasAddР
Conv1_transpose/ReluRelu Conv1_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:          2
Conv1_transpose/ReluА
Conv2_transpose/ShapeShape"Conv1_transpose/Relu:activations:0*
T0*
_output_shapes
:2
Conv2_transpose/ShapeФ
#Conv2_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Conv2_transpose/strided_slice/stackШ
%Conv2_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv2_transpose/strided_slice/stack_1Ш
%Conv2_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Conv2_transpose/strided_slice/stack_2┬
Conv2_transpose/strided_sliceStridedSliceConv2_transpose/Shape:output:0,Conv2_transpose/strided_slice/stack:output:0.Conv2_transpose/strided_slice/stack_1:output:0.Conv2_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Conv2_transpose/strided_slicet
Conv2_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
Conv2_transpose/stack/1t
Conv2_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
Conv2_transpose/stack/2t
Conv2_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Conv2_transpose/stack/3Є
Conv2_transpose/stackPack&Conv2_transpose/strided_slice:output:0 Conv2_transpose/stack/1:output:0 Conv2_transpose/stack/2:output:0 Conv2_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
Conv2_transpose/stackШ
%Conv2_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%Conv2_transpose/strided_slice_1/stackЬ
'Conv2_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2_transpose/strided_slice_1/stack_1Ь
'Conv2_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'Conv2_transpose/strided_slice_1/stack_2╠
Conv2_transpose/strided_slice_1StridedSliceConv2_transpose/stack:output:0.Conv2_transpose/strided_slice_1/stack:output:00Conv2_transpose/strided_slice_1/stack_1:output:00Conv2_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
Conv2_transpose/strided_slice_1у
/Conv2_transpose/conv2d_transpose/ReadVariableOpReadVariableOp8conv2_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype021
/Conv2_transpose/conv2d_transpose/ReadVariableOp║
 Conv2_transpose/conv2d_transposeConv2DBackpropInputConv2_transpose/stack:output:07Conv2_transpose/conv2d_transpose/ReadVariableOp:value:0"Conv1_transpose/Relu:activations:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2"
 Conv2_transpose/conv2d_transpose╝
&Conv2_transpose/BiasAdd/ReadVariableOpReadVariableOp/conv2_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&Conv2_transpose/BiasAdd/ReadVariableOp╥
Conv2_transpose/BiasAddBiasAdd)Conv2_transpose/conv2d_transpose:output:0.Conv2_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
Conv2_transpose/BiasAddР
Conv2_transpose/ReluRelu Conv2_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:          2
Conv2_transpose/Relu╝
"output_layer/Conv2D/ReadVariableOpReadVariableOp+output_layer_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"output_layer/Conv2D/ReadVariableOpц
output_layer/Conv2DConv2D"Conv2_transpose/Relu:activations:0*output_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
output_layer/Conv2D│
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOp╝
output_layer/BiasAddBiasAddoutput_layer/Conv2D:output:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
output_layer/BiasAddР
output_layer/SigmoidSigmoidoutput_layer/BiasAdd:output:0*
T0*/
_output_shapes
:         2
output_layer/Sigmoidя
IdentityIdentityoutput_layer/Sigmoid:y:0^Conv1/BiasAdd/ReadVariableOp^Conv1/Conv2D/ReadVariableOp'^Conv1_transpose/BiasAdd/ReadVariableOp0^Conv1_transpose/conv2d_transpose/ReadVariableOp^Conv2/BiasAdd/ReadVariableOp^Conv2/Conv2D/ReadVariableOp'^Conv2_transpose/BiasAdd/ReadVariableOp0^Conv2_transpose/conv2d_transpose/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2<
Conv1/BiasAdd/ReadVariableOpConv1/BiasAdd/ReadVariableOp2:
Conv1/Conv2D/ReadVariableOpConv1/Conv2D/ReadVariableOp2P
&Conv1_transpose/BiasAdd/ReadVariableOp&Conv1_transpose/BiasAdd/ReadVariableOp2b
/Conv1_transpose/conv2d_transpose/ReadVariableOp/Conv1_transpose/conv2d_transpose/ReadVariableOp2<
Conv2/BiasAdd/ReadVariableOpConv2/BiasAdd/ReadVariableOp2:
Conv2/Conv2D/ReadVariableOpConv2/Conv2D/ReadVariableOp2P
&Conv2_transpose/BiasAdd/ReadVariableOp&Conv2_transpose/BiasAdd/ReadVariableOp2b
/Conv2_transpose/conv2d_transpose/ReadVariableOp/Conv2_transpose/conv2d_transpose/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/Conv2D/ReadVariableOp"output_layer/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Р
∙
@__inference_Conv1_layer_call_and_return_conditional_losses_50959

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Р
∙
@__inference_Conv1_layer_call_and_return_conditional_losses_50457

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
П

С
#__inference_signature_wrapper_50764
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_503252
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
т$
з
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50700
input_1%
conv1_50672: 
conv1_50674: %
conv2_50678:  
conv2_50680: /
conv1_transpose_50684:  #
conv1_transpose_50686: /
conv2_transpose_50689:  #
conv2_transpose_50691: ,
output_layer_50694:  
output_layer_50696:
identityИвConv1/StatefulPartitionedCallв'Conv1_transpose/StatefulPartitionedCallвConv2/StatefulPartitionedCallв'Conv2_transpose/StatefulPartitionedCallв$output_layer/StatefulPartitionedCallО
Conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_50672conv1_50674*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv1_layer_call_and_return_conditional_losses_504572
Conv1/StatefulPartitionedCallї
Pool1/PartitionedCallPartitionedCall&Conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool1_layer_call_and_return_conditional_losses_503312
Pool1/PartitionedCallе
Conv2/StatefulPartitionedCallStatefulPartitionedCallPool1/PartitionedCall:output:0conv2_50678conv2_50680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Conv2_layer_call_and_return_conditional_losses_504752
Conv2/StatefulPartitionedCallї
Pool2/PartitionedCallPartitionedCall&Conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool2_layer_call_and_return_conditional_losses_503432
Pool2/PartitionedCallщ
'Conv1_transpose/StatefulPartitionedCallStatefulPartitionedCallPool2/PartitionedCall:output:0conv1_transpose_50684conv1_transpose_50686*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_503842)
'Conv1_transpose/StatefulPartitionedCall√
'Conv2_transpose/StatefulPartitionedCallStatefulPartitionedCall0Conv1_transpose/StatefulPartitionedCall:output:0conv2_transpose_50689conv2_transpose_50691*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_504292)
'Conv2_transpose/StatefulPartitionedCallь
$output_layer/StatefulPartitionedCallStatefulPartitionedCall0Conv2_transpose/StatefulPartitionedCall:output:0output_layer_50694output_layer_50696*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_505032&
$output_layer/StatefulPartitionedCall╓
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^Conv1/StatefulPartitionedCall(^Conv1_transpose/StatefulPartitionedCall^Conv2/StatefulPartitionedCall(^Conv2_transpose/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2R
'Conv1_transpose/StatefulPartitionedCall'Conv1_transpose/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2R
'Conv2_transpose/StatefulPartitionedCall'Conv2_transpose/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
─
A
%__inference_Pool2_layer_call_fn_50349

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_Pool2_layer_call_and_return_conditional_losses_503432
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Р
∙
@__inference_Conv2_layer_call_and_return_conditional_losses_50475

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Р
∙
@__inference_Conv2_layer_call_and_return_conditional_losses_50979

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Я
\
@__inference_Pool2_layer_call_and_return_conditional_losses_50343

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
serving_defaultл
C
input_18
serving_default_input_1:0         H
output_layer8
StatefulPartitionedCall:0         tensorflow/serving/predict:ое
фZ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
*{&call_and_return_all_conditional_losses
|__call__
}_default_save_signature"пW
_tf_keras_networkУW{"name": "AutoEncoder-Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "AutoEncoder-Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Pool1", "inbound_nodes": [[["Conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv2", "inbound_nodes": [[["Pool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "Pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Pool2", "inbound_nodes": [[["Conv2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv1_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv1_transpose", "inbound_nodes": [[["Pool2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv2_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv2_transpose", "inbound_nodes": [[["Conv1_transpose", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["Conv2_transpose", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 18, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "AutoEncoder-Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling2D", "config": {"name": "Pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Pool1", "inbound_nodes": [[["Conv1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "Conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv2", "inbound_nodes": [[["Pool1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "MaxPooling2D", "config": {"name": "Pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "Pool2", "inbound_nodes": [[["Conv2", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv1_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv1_transpose", "inbound_nodes": [[["Pool2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Conv2DTranspose", "config": {"name": "Conv2_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Conv2_transpose", "inbound_nodes": [[["Conv1_transpose", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv2D", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["Conv2_transpose", 0, 0, {}]]], "shared_object_id": 17}], "input_layers": [["input_1", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
∙"Ў
_tf_keras_input_layer╓{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
є


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*~&call_and_return_all_conditional_losses
__call__"╬	
_tf_keras_layer┤	{"name": "Conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
┼
trainable_variables
	variables
regularization_losses
	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"┤
_tf_keras_layerЪ{"name": "Pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "Pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["Conv1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}}
ї


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"╬	
_tf_keras_layer┤	{"name": "Conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Pool1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
┼
trainable_variables
 	variables
!regularization_losses
"	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"┤
_tf_keras_layerЪ{"name": "Pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "Pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["Conv2", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 23}}
к

#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"Г

_tf_keras_layerщ	{"name": "Conv1_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "Conv1_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["Pool2", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
╖

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"Р

_tf_keras_layerЎ	{"name": "Conv2_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "Conv2_transpose", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["Conv1_transpose", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
Т

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"ы	
_tf_keras_layer╤	{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Conv2_transpose", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
З
5iter

6beta_1

7beta_2
	8decay
9learning_ratemgmhmimj#mk$ml)mm*mn/mo0mpvqvrvsvt#vu$vv)vw*vx/vy0vz"
	optimizer
f
0
1
2
3
#4
$5
)6
*7
/8
09"
trackable_list_wrapper
f
0
1
2
3
#4
$5
)6
*7
/8
09"
trackable_list_wrapper
 "
trackable_list_wrapper
╩

trainable_variables
:layer_regularization_losses
;metrics

<layers
=non_trainable_variables
>layer_metrics
	variables
regularization_losses
|__call__
}_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
-
Мserving_default"
signature_map
&:$ 2Conv1/kernel
: 2
Conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
trainable_variables
?layer_regularization_losses
@metrics

Alayers
Bnon_trainable_variables
Clayer_metrics
	variables
regularization_losses
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
trainable_variables
Dlayer_regularization_losses
Emetrics

Flayers
Gnon_trainable_variables
Hlayer_metrics
	variables
regularization_losses
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
&:$  2Conv2/kernel
: 2
Conv2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
trainable_variables
Ilayer_regularization_losses
Jmetrics

Klayers
Lnon_trainable_variables
Mlayer_metrics
	variables
regularization_losses
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
trainable_variables
Nlayer_regularization_losses
Ometrics

Players
Qnon_trainable_variables
Rlayer_metrics
 	variables
!regularization_losses
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
0:.  2Conv1_transpose/kernel
":  2Conv1_transpose/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
%trainable_variables
Slayer_regularization_losses
Tmetrics

Ulayers
Vnon_trainable_variables
Wlayer_metrics
&	variables
'regularization_losses
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
0:.  2Conv2_transpose/kernel
":  2Conv2_transpose/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
+trainable_variables
Xlayer_regularization_losses
Ymetrics

Zlayers
[non_trainable_variables
\layer_metrics
,	variables
-regularization_losses
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
-:+ 2output_layer/kernel
:2output_layer/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
░
1trainable_variables
]layer_regularization_losses
^metrics

_layers
`non_trainable_variables
alayer_metrics
2	variables
3regularization_losses
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
'
b0"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╘
	ctotal
	dcount
e	variables
f	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 27}
:  (2total
:  (2count
.
c0
d1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
+:) 2Adam/Conv1/kernel/m
: 2Adam/Conv1/bias/m
+:)  2Adam/Conv2/kernel/m
: 2Adam/Conv2/bias/m
5:3  2Adam/Conv1_transpose/kernel/m
':% 2Adam/Conv1_transpose/bias/m
5:3  2Adam/Conv2_transpose/kernel/m
':% 2Adam/Conv2_transpose/bias/m
2:0 2Adam/output_layer/kernel/m
$:"2Adam/output_layer/bias/m
+:) 2Adam/Conv1/kernel/v
: 2Adam/Conv1/bias/v
+:)  2Adam/Conv2/kernel/v
: 2Adam/Conv2/bias/v
5:3  2Adam/Conv1_transpose/kernel/v
':% 2Adam/Conv1_transpose/bias/v
5:3  2Adam/Conv2_transpose/kernel/v
':% 2Adam/Conv2_transpose/bias/v
2:0 2Adam/output_layer/kernel/v
$:"2Adam/output_layer/bias/v
■2√
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50831
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50898
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50700
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50731└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
1__inference_AutoEncoder-Model_layer_call_fn_50533
1__inference_AutoEncoder-Model_layer_call_fn_50923
1__inference_AutoEncoder-Model_layer_call_fn_50948
1__inference_AutoEncoder-Model_layer_call_fn_50669└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ц2у
 __inference__wrapped_model_50325╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         
ъ2ч
@__inference_Conv1_layer_call_and_return_conditional_losses_50959в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_Conv1_layer_call_fn_50968в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
@__inference_Pool1_layer_call_and_return_conditional_losses_50331р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Н2К
%__inference_Pool1_layer_call_fn_50337р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
ъ2ч
@__inference_Conv2_layer_call_and_return_conditional_losses_50979в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_Conv2_layer_call_fn_50988в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
@__inference_Pool2_layer_call_and_return_conditional_losses_50343р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Н2К
%__inference_Pool2_layer_call_fn_50349р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
й2ж
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_50384╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
О2Л
/__inference_Conv1_transpose_layer_call_fn_50394╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
й2ж
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_50429╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
О2Л
/__inference_Conv2_transpose_layer_call_fn_50439╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
ё2ю
G__inference_output_layer_layer_call_and_return_conditional_losses_50999в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_output_layer_layer_call_fn_51008в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╩B╟
#__inference_signature_wrapper_50764input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 р
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50700П
#$)*/0@в=
6в3
)К&
input_1         
p 

 
к "?в<
5К2
0+                           
Ъ р
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50731П
#$)*/0@в=
6в3
)К&
input_1         
p

 
к "?в<
5К2
0+                           
Ъ ╠
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50831|
#$)*/0?в<
5в2
(К%
inputs         
p 

 
к "-в*
#К 
0         
Ъ ╠
L__inference_AutoEncoder-Model_layer_call_and_return_conditional_losses_50898|
#$)*/0?в<
5в2
(К%
inputs         
p

 
к "-в*
#К 
0         
Ъ ╕
1__inference_AutoEncoder-Model_layer_call_fn_50533В
#$)*/0@в=
6в3
)К&
input_1         
p 

 
к "2К/+                           ╕
1__inference_AutoEncoder-Model_layer_call_fn_50669В
#$)*/0@в=
6в3
)К&
input_1         
p

 
к "2К/+                           ╖
1__inference_AutoEncoder-Model_layer_call_fn_50923Б
#$)*/0?в<
5в2
(К%
inputs         
p 

 
к "2К/+                           ╖
1__inference_AutoEncoder-Model_layer_call_fn_50948Б
#$)*/0?в<
5в2
(К%
inputs         
p

 
к "2К/+                           ░
@__inference_Conv1_layer_call_and_return_conditional_losses_50959l7в4
-в*
(К%
inputs         
к "-в*
#К 
0          
Ъ И
%__inference_Conv1_layer_call_fn_50968_7в4
-в*
(К%
inputs         
к " К          ▀
J__inference_Conv1_transpose_layer_call_and_return_conditional_losses_50384Р#$IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ ╖
/__inference_Conv1_transpose_layer_call_fn_50394Г#$IвF
?в<
:К7
inputs+                            
к "2К/+                            ░
@__inference_Conv2_layer_call_and_return_conditional_losses_50979l7в4
-в*
(К%
inputs          
к "-в*
#К 
0          
Ъ И
%__inference_Conv2_layer_call_fn_50988_7в4
-в*
(К%
inputs          
к " К          ▀
J__inference_Conv2_transpose_layer_call_and_return_conditional_losses_50429Р)*IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ ╖
/__inference_Conv2_transpose_layer_call_fn_50439Г)*IвF
?в<
:К7
inputs+                            
к "2К/+                            у
@__inference_Pool1_layer_call_and_return_conditional_losses_50331ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╗
%__inference_Pool1_layer_call_fn_50337СRвO
HвE
CК@
inputs4                                    
к ";К84                                    у
@__inference_Pool2_layer_call_and_return_conditional_losses_50343ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╗
%__inference_Pool2_layer_call_fn_50349СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ░
 __inference__wrapped_model_50325Л
#$)*/08в5
.в+
)К&
input_1         
к "Cк@
>
output_layer.К+
output_layer         ▄
G__inference_output_layer_layer_call_and_return_conditional_losses_50999Р/0IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           
Ъ ┤
,__inference_output_layer_layer_call_fn_51008Г/0IвF
?в<
:К7
inputs+                            
к "2К/+                           ╛
#__inference_signature_wrapper_50764Ц
#$)*/0Cв@
в 
9к6
4
input_1)К&
input_1         "Cк@
>
output_layer.К+
output_layer         