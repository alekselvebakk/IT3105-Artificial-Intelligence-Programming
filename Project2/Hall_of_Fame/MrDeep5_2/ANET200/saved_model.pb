��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
{
hidden_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%�* 
shared_namehidden_0/kernel
t
#hidden_0/kernel/Read/ReadVariableOpReadVariableOphidden_0/kernel*
_output_shapes
:	%�*
dtype0
s
hidden_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namehidden_0/bias
l
!hidden_0/bias/Read/ReadVariableOpReadVariableOphidden_0/bias*
_output_shapes	
:�*
dtype0
|
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namehidden_1/kernel
u
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel* 
_output_shapes
:
��*
dtype0
s
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namehidden_1/bias
l
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes	
:�*
dtype0
{
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d* 
shared_namehidden_2/kernel
t
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel*
_output_shapes
:	�d*
dtype0
r
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namehidden_2/bias
k
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes
:d*
dtype0
�
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d$*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:d$*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:$*
dtype0
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
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
�
#Adagrad/hidden_0/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%�*4
shared_name%#Adagrad/hidden_0/kernel/accumulator
�
7Adagrad/hidden_0/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_0/kernel/accumulator*
_output_shapes
:	%�*
dtype0
�
!Adagrad/hidden_0/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adagrad/hidden_0/bias/accumulator
�
5Adagrad/hidden_0/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_0/bias/accumulator*
_output_shapes	
:�*
dtype0
�
#Adagrad/hidden_1/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adagrad/hidden_1/kernel/accumulator
�
7Adagrad/hidden_1/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_1/kernel/accumulator* 
_output_shapes
:
��*
dtype0
�
!Adagrad/hidden_1/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adagrad/hidden_1/bias/accumulator
�
5Adagrad/hidden_1/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_1/bias/accumulator*
_output_shapes	
:�*
dtype0
�
#Adagrad/hidden_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*4
shared_name%#Adagrad/hidden_2/kernel/accumulator
�
7Adagrad/hidden_2/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_2/kernel/accumulator*
_output_shapes
:	�d*
dtype0
�
!Adagrad/hidden_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*2
shared_name#!Adagrad/hidden_2/bias/accumulator
�
5Adagrad/hidden_2/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_2/bias/accumulator*
_output_shapes
:d*
dtype0
�
'Adagrad/output_layer/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d$*8
shared_name)'Adagrad/output_layer/kernel/accumulator
�
;Adagrad/output_layer/kernel/accumulator/Read/ReadVariableOpReadVariableOp'Adagrad/output_layer/kernel/accumulator*
_output_shapes

:d$*
dtype0
�
%Adagrad/output_layer/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*6
shared_name'%Adagrad/output_layer/bias/accumulator
�
9Adagrad/output_layer/bias/accumulator/Read/ReadVariableOpReadVariableOp%Adagrad/output_layer/bias/accumulator*
_output_shapes
:$*
dtype0

NoOpNoOp
�&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
%
#_self_saveable_object_factories
�

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
�

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
�

kernel
bias
#_self_saveable_object_factories
	variables
 trainable_variables
!regularization_losses
"	keras_api
�

#kernel
$bias
#%_self_saveable_object_factories
&	variables
'trainable_variables
(regularization_losses
)	keras_api
�
*iter
	+decay
,learning_rateaccumulatorKaccumulatorLaccumulatorMaccumulatorNaccumulatorOaccumulatorP#accumulatorQ$accumulatorR
 
 
8
0
1
2
3
4
5
#6
$7
8
0
1
2
3
4
5
#6
$7
 
�

-layers
.non_trainable_variables
		variables
/layer_metrics

trainable_variables
0layer_regularization_losses
regularization_losses
1metrics
 
[Y
VARIABLE_VALUEhidden_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�

2layers
3non_trainable_variables
	variables
4layer_metrics
trainable_variables
5layer_regularization_losses
regularization_losses
6metrics
[Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�

7layers
8non_trainable_variables
	variables
9layer_metrics
trainable_variables
:layer_regularization_losses
regularization_losses
;metrics
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
�

<layers
=non_trainable_variables
	variables
>layer_metrics
 trainable_variables
?layer_regularization_losses
!regularization_losses
@metrics
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
 
�

Alayers
Bnon_trainable_variables
&	variables
Clayer_metrics
'trainable_variables
Dlayer_regularization_losses
(regularization_losses
Emetrics
KI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4
 
 
 

F0
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
	Gtotal
	Hcount
I	variables
J	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

I	variables
��
VARIABLE_VALUE#Adagrad/hidden_0/kernel/accumulator\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adagrad/hidden_0/bias/accumulatorZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adagrad/hidden_1/kernel/accumulator\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adagrad/hidden_1/bias/accumulatorZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adagrad/hidden_2/kernel/accumulator\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adagrad/hidden_2/bias/accumulatorZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'Adagrad/output_layer/kernel/accumulator\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Adagrad/output_layer/bias/accumulatorZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_layerPlaceholder*'
_output_shapes
:���������%*
dtype0*
shape:���������%
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerhidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biasoutput_layer/kerneloutput_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_222875495
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_0/kernel/Read/ReadVariableOp!hidden_0/bias/Read/ReadVariableOp#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adagrad/hidden_0/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_0/bias/accumulator/Read/ReadVariableOp7Adagrad/hidden_1/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_1/bias/accumulator/Read/ReadVariableOp7Adagrad/hidden_2/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_2/bias/accumulator/Read/ReadVariableOp;Adagrad/output_layer/kernel/accumulator/Read/ReadVariableOp9Adagrad/output_layer/bias/accumulator/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_save_222875767
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biasoutput_layer/kerneloutput_layer/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotalcount#Adagrad/hidden_0/kernel/accumulator!Adagrad/hidden_0/bias/accumulator#Adagrad/hidden_1/kernel/accumulator!Adagrad/hidden_1/bias/accumulator#Adagrad/hidden_2/kernel/accumulator!Adagrad/hidden_2/bias/accumulator'Adagrad/output_layer/kernel/accumulator%Adagrad/output_layer/bias/accumulator*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference__traced_restore_222875840��
�	
�
G__inference_hidden_0_layer_call_and_return_conditional_losses_222875255

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�	
�
K__inference_output_layer_layer_call_and_return_conditional_losses_222875336

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������$2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_model_layer_call_fn_222875601

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_2228754492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�	
�
G__inference_hidden_2_layer_call_and_return_conditional_losses_222875309

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_model_layer_call_and_return_conditional_losses_222875449

inputs
hidden_0_222875428
hidden_0_222875430
hidden_1_222875433
hidden_1_222875435
hidden_2_222875438
hidden_2_222875440
output_layer_222875443
output_layer_222875445
identity�� hidden_0/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall� hidden_2/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_222875428hidden_0_222875430*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_2228752552"
 hidden_0/StatefulPartitionedCall�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_222875433hidden_1_222875435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_2228752822"
 hidden_1/StatefulPartitionedCall�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_222875438hidden_2_222875440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_2228753092"
 hidden_2/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0output_layer_222875443output_layer_222875445*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_2228753362&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
,__inference_hidden_1_layer_call_fn_222875641

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_2228752822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�]
�
%__inference__traced_restore_222875840
file_prefix$
 assignvariableop_hidden_0_kernel$
 assignvariableop_1_hidden_0_bias&
"assignvariableop_2_hidden_1_kernel$
 assignvariableop_3_hidden_1_bias&
"assignvariableop_4_hidden_2_kernel$
 assignvariableop_5_hidden_2_bias*
&assignvariableop_6_output_layer_kernel(
$assignvariableop_7_output_layer_bias#
assignvariableop_8_adagrad_iter$
 assignvariableop_9_adagrad_decay-
)assignvariableop_10_adagrad_learning_rate
assignvariableop_11_total
assignvariableop_12_count;
7assignvariableop_13_adagrad_hidden_0_kernel_accumulator9
5assignvariableop_14_adagrad_hidden_0_bias_accumulator;
7assignvariableop_15_adagrad_hidden_1_kernel_accumulator9
5assignvariableop_16_adagrad_hidden_1_bias_accumulator;
7assignvariableop_17_adagrad_hidden_2_kernel_accumulator9
5assignvariableop_18_adagrad_hidden_2_bias_accumulator?
;assignvariableop_19_adagrad_output_layer_kernel_accumulator=
9assignvariableop_20_adagrad_output_layer_bias_accumulator
identity_22��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_hidden_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_hidden_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_hidden_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_output_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_output_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adagrad_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_adagrad_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_adagrad_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_adagrad_hidden_0_kernel_accumulatorIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp5assignvariableop_14_adagrad_hidden_0_bias_accumulatorIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adagrad_hidden_1_kernel_accumulatorIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adagrad_hidden_1_bias_accumulatorIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adagrad_hidden_2_kernel_accumulatorIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adagrad_hidden_2_bias_accumulatorIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp;assignvariableop_19_adagrad_output_layer_kernel_accumulatorIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp9assignvariableop_20_adagrad_output_layer_bias_accumulatorIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21�
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
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
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
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
�	
�
G__inference_hidden_1_layer_call_and_return_conditional_losses_222875632

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_output_layer_layer_call_fn_222875681

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_2228753362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
G__inference_hidden_0_layer_call_and_return_conditional_losses_222875612

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�6
�	
"__inference__traced_save_222875767
file_prefix.
*savev2_hidden_0_kernel_read_readvariableop,
(savev2_hidden_0_bias_read_readvariableop.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_adagrad_hidden_0_kernel_accumulator_read_readvariableop@
<savev2_adagrad_hidden_0_bias_accumulator_read_readvariableopB
>savev2_adagrad_hidden_1_kernel_accumulator_read_readvariableop@
<savev2_adagrad_hidden_1_bias_accumulator_read_readvariableopB
>savev2_adagrad_hidden_2_kernel_accumulator_read_readvariableop@
<savev2_adagrad_hidden_2_bias_accumulator_read_readvariableopF
Bsavev2_adagrad_output_layer_kernel_accumulator_read_readvariableopD
@savev2_adagrad_output_layer_bias_accumulator_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_0_kernel_read_readvariableop(savev2_hidden_0_bias_read_readvariableop*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adagrad_hidden_0_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_0_bias_accumulator_read_readvariableop>savev2_adagrad_hidden_1_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_1_bias_accumulator_read_readvariableop>savev2_adagrad_hidden_2_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_2_bias_accumulator_read_readvariableopBsavev2_adagrad_output_layer_kernel_accumulator_read_readvariableop@savev2_adagrad_output_layer_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	%�:�:
��:�:	�d:d:d$:$: : : : : :	%�:�:
��:�:	�d:d:d$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	%�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:d$: 

_output_shapes
:$:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	%�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:d$: 

_output_shapes
:$:

_output_shapes
: 
�
�
,__inference_hidden_2_layer_call_fn_222875661

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_2228753092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
$__inference__wrapped_model_222875240
input_layer1
-model_hidden_0_matmul_readvariableop_resource2
.model_hidden_0_biasadd_readvariableop_resource1
-model_hidden_1_matmul_readvariableop_resource2
.model_hidden_1_biasadd_readvariableop_resource1
-model_hidden_2_matmul_readvariableop_resource2
.model_hidden_2_biasadd_readvariableop_resource5
1model_output_layer_matmul_readvariableop_resource6
2model_output_layer_biasadd_readvariableop_resource
identity��%model/hidden_0/BiasAdd/ReadVariableOp�$model/hidden_0/MatMul/ReadVariableOp�%model/hidden_1/BiasAdd/ReadVariableOp�$model/hidden_1/MatMul/ReadVariableOp�%model/hidden_2/BiasAdd/ReadVariableOp�$model/hidden_2/MatMul/ReadVariableOp�)model/output_layer/BiasAdd/ReadVariableOp�(model/output_layer/MatMul/ReadVariableOp�
$model/hidden_0/MatMul/ReadVariableOpReadVariableOp-model_hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%�*
dtype02&
$model/hidden_0/MatMul/ReadVariableOp�
model/hidden_0/MatMulMatMulinput_layer,model/hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/hidden_0/MatMul�
%model/hidden_0/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%model/hidden_0/BiasAdd/ReadVariableOp�
model/hidden_0/BiasAddBiasAddmodel/hidden_0/MatMul:product:0-model/hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/hidden_0/BiasAdd�
model/hidden_0/ReluRelumodel/hidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model/hidden_0/Relu�
$model/hidden_1/MatMul/ReadVariableOpReadVariableOp-model_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02&
$model/hidden_1/MatMul/ReadVariableOp�
model/hidden_1/MatMulMatMul!model/hidden_0/Relu:activations:0,model/hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/hidden_1/MatMul�
%model/hidden_1/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%model/hidden_1/BiasAdd/ReadVariableOp�
model/hidden_1/BiasAddBiasAddmodel/hidden_1/MatMul:product:0-model/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/hidden_1/BiasAdd�
model/hidden_1/ReluRelumodel/hidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model/hidden_1/Relu�
$model/hidden_2/MatMul/ReadVariableOpReadVariableOp-model_hidden_2_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02&
$model/hidden_2/MatMul/ReadVariableOp�
model/hidden_2/MatMulMatMul!model/hidden_1/Relu:activations:0,model/hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/hidden_2/MatMul�
%model/hidden_2/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02'
%model/hidden_2/BiasAdd/ReadVariableOp�
model/hidden_2/BiasAddBiasAddmodel/hidden_2/MatMul:product:0-model/hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/hidden_2/BiasAdd�
model/hidden_2/ReluRelumodel/hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
model/hidden_2/Relu�
(model/output_layer/MatMul/ReadVariableOpReadVariableOp1model_output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02*
(model/output_layer/MatMul/ReadVariableOp�
model/output_layer/MatMulMatMul!model/hidden_2/Relu:activations:00model/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
model/output_layer/MatMul�
)model/output_layer/BiasAdd/ReadVariableOpReadVariableOp2model_output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02+
)model/output_layer/BiasAdd/ReadVariableOp�
model/output_layer/BiasAddBiasAdd#model/output_layer/MatMul:product:01model/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
model/output_layer/BiasAdd�
model/output_layer/SoftmaxSoftmax#model/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������$2
model/output_layer/Softmax�
IdentityIdentity$model/output_layer/Softmax:softmax:0&^model/hidden_0/BiasAdd/ReadVariableOp%^model/hidden_0/MatMul/ReadVariableOp&^model/hidden_1/BiasAdd/ReadVariableOp%^model/hidden_1/MatMul/ReadVariableOp&^model/hidden_2/BiasAdd/ReadVariableOp%^model/hidden_2/MatMul/ReadVariableOp*^model/output_layer/BiasAdd/ReadVariableOp)^model/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::2N
%model/hidden_0/BiasAdd/ReadVariableOp%model/hidden_0/BiasAdd/ReadVariableOp2L
$model/hidden_0/MatMul/ReadVariableOp$model/hidden_0/MatMul/ReadVariableOp2N
%model/hidden_1/BiasAdd/ReadVariableOp%model/hidden_1/BiasAdd/ReadVariableOp2L
$model/hidden_1/MatMul/ReadVariableOp$model/hidden_1/MatMul/ReadVariableOp2N
%model/hidden_2/BiasAdd/ReadVariableOp%model/hidden_2/BiasAdd/ReadVariableOp2L
$model/hidden_2/MatMul/ReadVariableOp$model/hidden_2/MatMul/ReadVariableOp2V
)model/output_layer/BiasAdd/ReadVariableOp)model/output_layer/BiasAdd/ReadVariableOp2T
(model/output_layer/MatMul/ReadVariableOp(model/output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
D__inference_model_layer_call_and_return_conditional_losses_222875404

inputs
hidden_0_222875383
hidden_0_222875385
hidden_1_222875388
hidden_1_222875390
hidden_2_222875393
hidden_2_222875395
output_layer_222875398
output_layer_222875400
identity�� hidden_0/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall� hidden_2/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_222875383hidden_0_222875385*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_2228752552"
 hidden_0/StatefulPartitionedCall�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_222875388hidden_1_222875390*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_2228752822"
 hidden_1/StatefulPartitionedCall�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_222875393hidden_2_222875395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_2228753092"
 hidden_2/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0output_layer_222875398output_layer_222875400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_2228753362&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�	
�
K__inference_output_layer_layer_call_and_return_conditional_losses_222875672

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������$2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�'
�
D__inference_model_layer_call_and_return_conditional_losses_222875527

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource+
'hidden_1_matmul_readvariableop_resource,
(hidden_1_biasadd_readvariableop_resource+
'hidden_2_matmul_readvariableop_resource,
(hidden_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��hidden_0/BiasAdd/ReadVariableOp�hidden_0/MatMul/ReadVariableOp�hidden_1/BiasAdd/ReadVariableOp�hidden_1/MatMul/ReadVariableOp�hidden_2/BiasAdd/ReadVariableOp�hidden_2/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%�*
dtype02 
hidden_0/MatMul/ReadVariableOp�
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_0/MatMul�
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
hidden_0/BiasAdd/ReadVariableOp�
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
hidden_0/Relu�
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
hidden_1/MatMul/ReadVariableOp�
hidden_1/MatMulMatMulhidden_0/Relu:activations:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_1/MatMul�
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
hidden_1/BiasAdd/ReadVariableOp�
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
hidden_1/Relu�
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02 
hidden_2/MatMul/ReadVariableOp�
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
hidden_2/MatMul�
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
hidden_2/BiasAdd/ReadVariableOp�
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
hidden_2/BiasAdds
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
hidden_2/Relu�
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMulhidden_2/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
output_layer/MatMul�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#output_layer/BiasAdd/ReadVariableOp�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
output_layer/BiasAdd�
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������$2
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�	
�
G__inference_hidden_1_layer_call_and_return_conditional_losses_222875282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
G__inference_hidden_2_layer_call_and_return_conditional_losses_222875652

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_model_layer_call_fn_222875580

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_2228754042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
D__inference_model_layer_call_and_return_conditional_losses_222875353
input_layer
hidden_0_222875266
hidden_0_222875268
hidden_1_222875293
hidden_1_222875295
hidden_2_222875320
hidden_2_222875322
output_layer_222875347
output_layer_222875349
identity�� hidden_0/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall� hidden_2/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_222875266hidden_0_222875268*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_2228752552"
 hidden_0/StatefulPartitionedCall�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_222875293hidden_1_222875295*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_2228752822"
 hidden_1/StatefulPartitionedCall�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_222875320hidden_2_222875322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_2228753092"
 hidden_2/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0output_layer_222875347output_layer_222875349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_2228753362&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
)__inference_model_layer_call_fn_222875468
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_2228754492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
,__inference_hidden_0_layer_call_fn_222875621

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_2228752552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������%::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
D__inference_model_layer_call_and_return_conditional_losses_222875377
input_layer
hidden_0_222875356
hidden_0_222875358
hidden_1_222875361
hidden_1_222875363
hidden_2_222875366
hidden_2_222875368
output_layer_222875371
output_layer_222875373
identity�� hidden_0/StatefulPartitionedCall� hidden_1/StatefulPartitionedCall� hidden_2/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_222875356hidden_0_222875358*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_2228752552"
 hidden_0/StatefulPartitionedCall�
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_222875361hidden_1_222875363*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_2228752822"
 hidden_1/StatefulPartitionedCall�
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_222875366hidden_2_222875368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_2228753092"
 hidden_2/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0output_layer_222875371output_layer_222875373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_2228753362&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
'__inference_signature_wrapper_222875495
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_2228752402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�'
�
D__inference_model_layer_call_and_return_conditional_losses_222875559

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource+
'hidden_1_matmul_readvariableop_resource,
(hidden_1_biasadd_readvariableop_resource+
'hidden_2_matmul_readvariableop_resource,
(hidden_2_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��hidden_0/BiasAdd/ReadVariableOp�hidden_0/MatMul/ReadVariableOp�hidden_1/BiasAdd/ReadVariableOp�hidden_1/MatMul/ReadVariableOp�hidden_2/BiasAdd/ReadVariableOp�hidden_2/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%�*
dtype02 
hidden_0/MatMul/ReadVariableOp�
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_0/MatMul�
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
hidden_0/BiasAdd/ReadVariableOp�
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
hidden_0/Relu�
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
hidden_1/MatMul/ReadVariableOp�
hidden_1/MatMulMatMulhidden_0/Relu:activations:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_1/MatMul�
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
hidden_1/BiasAdd/ReadVariableOp�
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
hidden_1/Relu�
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02 
hidden_2/MatMul/ReadVariableOp�
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
hidden_2/MatMul�
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
hidden_2/BiasAdd/ReadVariableOp�
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
hidden_2/BiasAdds
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
hidden_2/Relu�
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMulhidden_2/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
output_layer/MatMul�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#output_layer/BiasAdd/ReadVariableOp�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
output_layer/BiasAdd�
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������$2
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
)__inference_model_layer_call_fn_222875423
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_2228754042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������%::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_layer4
serving_default_input_layer:0���������%@
output_layer0
StatefulPartitionedCall:0���������$tensorflow/serving/predict:�
�.
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
S__call__
T_default_save_signature
*U&call_and_return_all_conditional_losses"�+
_tf_keras_network�+{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_1", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_2", "inbound_nodes": [[["hidden_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 37]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_1", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_2", "inbound_nodes": [[["hidden_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "cross_entropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.014999999664723873, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
�
#_self_saveable_object_factories"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
�

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}}
�

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
�

kernel
bias
#_self_saveable_object_factories
	variables
 trainable_variables
!regularization_losses
"	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
�

#kernel
$bias
#%_self_saveable_object_factories
&	variables
'trainable_variables
(regularization_losses
)	keras_api
\__call__
*]&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
*iter
	+decay
,learning_rateaccumulatorKaccumulatorLaccumulatorMaccumulatorNaccumulatorOaccumulatorP#accumulatorQ$accumulatorR"
	optimizer
,
^serving_default"
signature_map
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
 "
trackable_list_wrapper
�

-layers
.non_trainable_variables
		variables
/layer_metrics

trainable_variables
0layer_regularization_losses
regularization_losses
1metrics
S__call__
T_default_save_signature
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
": 	%�2hidden_0/kernel
:�2hidden_0/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

2layers
3non_trainable_variables
	variables
4layer_metrics
trainable_variables
5layer_regularization_losses
regularization_losses
6metrics
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
#:!
��2hidden_1/kernel
:�2hidden_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

7layers
8non_trainable_variables
	variables
9layer_metrics
trainable_variables
:layer_regularization_losses
regularization_losses
;metrics
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
": 	�d2hidden_2/kernel
:d2hidden_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

<layers
=non_trainable_variables
	variables
>layer_metrics
 trainable_variables
?layer_regularization_losses
!regularization_losses
@metrics
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
%:#d$2output_layer/kernel
:$2output_layer/bias
 "
trackable_dict_wrapper
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
�

Alayers
Bnon_trainable_variables
&	variables
Clayer_metrics
'trainable_variables
Dlayer_regularization_losses
(regularization_losses
Emetrics
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
F0"
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
�
	Gtotal
	Hcount
I	variables
J	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
G0
H1"
trackable_list_wrapper
-
I	variables"
_generic_user_object
4:2	%�2#Adagrad/hidden_0/kernel/accumulator
.:,�2!Adagrad/hidden_0/bias/accumulator
5:3
��2#Adagrad/hidden_1/kernel/accumulator
.:,�2!Adagrad/hidden_1/bias/accumulator
4:2	�d2#Adagrad/hidden_2/kernel/accumulator
-:+d2!Adagrad/hidden_2/bias/accumulator
7:5d$2'Adagrad/output_layer/kernel/accumulator
1:/$2%Adagrad/output_layer/bias/accumulator
�2�
)__inference_model_layer_call_fn_222875580
)__inference_model_layer_call_fn_222875601
)__inference_model_layer_call_fn_222875423
)__inference_model_layer_call_fn_222875468�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference__wrapped_model_222875240�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_layer���������%
�2�
D__inference_model_layer_call_and_return_conditional_losses_222875353
D__inference_model_layer_call_and_return_conditional_losses_222875527
D__inference_model_layer_call_and_return_conditional_losses_222875559
D__inference_model_layer_call_and_return_conditional_losses_222875377�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_hidden_0_layer_call_fn_222875621�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_hidden_0_layer_call_and_return_conditional_losses_222875612�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_hidden_1_layer_call_fn_222875641�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_hidden_1_layer_call_and_return_conditional_losses_222875632�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_hidden_2_layer_call_fn_222875661�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_hidden_2_layer_call_and_return_conditional_losses_222875652�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_output_layer_layer_call_fn_222875681�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_output_layer_layer_call_and_return_conditional_losses_222875672�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_222875495input_layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
$__inference__wrapped_model_222875240}#$4�1
*�'
%�"
input_layer���������%
� ";�8
6
output_layer&�#
output_layer���������$�
G__inference_hidden_0_layer_call_and_return_conditional_losses_222875612]/�,
%�"
 �
inputs���������%
� "&�#
�
0����������
� �
,__inference_hidden_0_layer_call_fn_222875621P/�,
%�"
 �
inputs���������%
� "������������
G__inference_hidden_1_layer_call_and_return_conditional_losses_222875632^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
,__inference_hidden_1_layer_call_fn_222875641Q0�-
&�#
!�
inputs����������
� "������������
G__inference_hidden_2_layer_call_and_return_conditional_losses_222875652]0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� �
,__inference_hidden_2_layer_call_fn_222875661P0�-
&�#
!�
inputs����������
� "����������d�
D__inference_model_layer_call_and_return_conditional_losses_222875353o#$<�9
2�/
%�"
input_layer���������%
p

 
� "%�"
�
0���������$
� �
D__inference_model_layer_call_and_return_conditional_losses_222875377o#$<�9
2�/
%�"
input_layer���������%
p 

 
� "%�"
�
0���������$
� �
D__inference_model_layer_call_and_return_conditional_losses_222875527j#$7�4
-�*
 �
inputs���������%
p

 
� "%�"
�
0���������$
� �
D__inference_model_layer_call_and_return_conditional_losses_222875559j#$7�4
-�*
 �
inputs���������%
p 

 
� "%�"
�
0���������$
� �
)__inference_model_layer_call_fn_222875423b#$<�9
2�/
%�"
input_layer���������%
p

 
� "����������$�
)__inference_model_layer_call_fn_222875468b#$<�9
2�/
%�"
input_layer���������%
p 

 
� "����������$�
)__inference_model_layer_call_fn_222875580]#$7�4
-�*
 �
inputs���������%
p

 
� "����������$�
)__inference_model_layer_call_fn_222875601]#$7�4
-�*
 �
inputs���������%
p 

 
� "����������$�
K__inference_output_layer_layer_call_and_return_conditional_losses_222875672\#$/�,
%�"
 �
inputs���������d
� "%�"
�
0���������$
� �
0__inference_output_layer_layer_call_fn_222875681O#$/�,
%�"
 �
inputs���������d
� "����������$�
'__inference_signature_wrapper_222875495�#$C�@
� 
9�6
4
input_layer%�"
input_layer���������%";�8
6
output_layer&�#
output_layer���������$