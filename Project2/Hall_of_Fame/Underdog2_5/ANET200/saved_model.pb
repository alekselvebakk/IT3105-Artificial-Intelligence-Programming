•Є
йЊ
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
Њ
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
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8Њг
{
hidden_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%Р* 
shared_namehidden_0/kernel
t
#hidden_0/kernel/Read/ReadVariableOpReadVariableOphidden_0/kernel*
_output_shapes
:	%Р*
dtype0
s
hidden_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*
shared_namehidden_0/bias
l
!hidden_0/bias/Read/ReadVariableOpReadVariableOphidden_0/bias*
_output_shapes	
:Р*
dtype0
|
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
РР* 
shared_namehidden_1/kernel
u
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel* 
_output_shapes
:
РР*
dtype0
s
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*
shared_namehidden_1/bias
l
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes	
:Р*
dtype0
|
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р»* 
shared_namehidden_2/kernel
u
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel* 
_output_shapes
:
Р»*
dtype0
s
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*
shared_namehidden_2/bias
l
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes	
:»*
dtype0
{
hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»d* 
shared_namehidden_3/kernel
t
#hidden_3/kernel/Read/ReadVariableOpReadVariableOphidden_3/kernel*
_output_shapes
:	»d*
dtype0
r
hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namehidden_3/bias
k
!hidden_3/bias/Read/ReadVariableOpReadVariableOphidden_3/bias*
_output_shapes
:d*
dtype0
В
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
£
#Adagrad/hidden_0/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%Р*4
shared_name%#Adagrad/hidden_0/kernel/accumulator
Ь
7Adagrad/hidden_0/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_0/kernel/accumulator*
_output_shapes
:	%Р*
dtype0
Ы
!Adagrad/hidden_0/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*2
shared_name#!Adagrad/hidden_0/bias/accumulator
Ф
5Adagrad/hidden_0/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_0/bias/accumulator*
_output_shapes	
:Р*
dtype0
§
#Adagrad/hidden_1/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
РР*4
shared_name%#Adagrad/hidden_1/kernel/accumulator
Э
7Adagrad/hidden_1/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_1/kernel/accumulator* 
_output_shapes
:
РР*
dtype0
Ы
!Adagrad/hidden_1/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р*2
shared_name#!Adagrad/hidden_1/bias/accumulator
Ф
5Adagrad/hidden_1/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_1/bias/accumulator*
_output_shapes	
:Р*
dtype0
§
#Adagrad/hidden_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р»*4
shared_name%#Adagrad/hidden_2/kernel/accumulator
Э
7Adagrad/hidden_2/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_2/kernel/accumulator* 
_output_shapes
:
Р»*
dtype0
Ы
!Adagrad/hidden_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*2
shared_name#!Adagrad/hidden_2/bias/accumulator
Ф
5Adagrad/hidden_2/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_2/bias/accumulator*
_output_shapes	
:»*
dtype0
£
#Adagrad/hidden_3/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»d*4
shared_name%#Adagrad/hidden_3/kernel/accumulator
Ь
7Adagrad/hidden_3/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_3/kernel/accumulator*
_output_shapes
:	»d*
dtype0
Ъ
!Adagrad/hidden_3/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*2
shared_name#!Adagrad/hidden_3/bias/accumulator
У
5Adagrad/hidden_3/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_3/bias/accumulator*
_output_shapes
:d*
dtype0
™
'Adagrad/output_layer/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d$*8
shared_name)'Adagrad/output_layer/kernel/accumulator
£
;Adagrad/output_layer/kernel/accumulator/Read/ReadVariableOpReadVariableOp'Adagrad/output_layer/kernel/accumulator*
_output_shapes

:d$*
dtype0
Ґ
%Adagrad/output_layer/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*6
shared_name'%Adagrad/output_layer/bias/accumulator
Ы
9Adagrad/output_layer/bias/accumulator/Read/ReadVariableOpReadVariableOp%Adagrad/output_layer/bias/accumulator*
_output_shapes
:$*
dtype0

NoOpNoOp
у-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ѓ-
value§-B°- BЪ-
ж
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer

signatures
#	_self_saveable_object_factories

	variables
regularization_losses
trainable_variables
	keras_api
%
#_self_saveable_object_factories
Н

kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
Н

kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
Н

kernel
bias
#_self_saveable_object_factories
 	variables
!regularization_losses
"trainable_variables
#	keras_api
Н

$kernel
%bias
#&_self_saveable_object_factories
'	variables
(regularization_losses
)trainable_variables
*	keras_api
Н

+kernel
,bias
#-_self_saveable_object_factories
.	variables
/regularization_losses
0trainable_variables
1	keras_api
ж
2iter
	3decay
4learning_rateaccumulatorXaccumulatorYaccumulatorZaccumulator[accumulator\accumulator]$accumulator^%accumulator_+accumulator`,accumulatora
 
 
F
0
1
2
3
4
5
$6
%7
+8
,9
 
F
0
1
2
3
4
5
$6
%7
+8
,9
≠
5layer_regularization_losses

	variables
regularization_losses

6layers
7non_trainable_variables
8layer_metrics
9metrics
trainable_variables
 
[Y
VARIABLE_VALUEhidden_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
≠
:layer_regularization_losses
	variables
regularization_losses

;layers
<non_trainable_variables
=layer_metrics
>metrics
trainable_variables
[Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
≠
?layer_regularization_losses
	variables
regularization_losses

@layers
Anon_trainable_variables
Blayer_metrics
Cmetrics
trainable_variables
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
≠
Dlayer_regularization_losses
 	variables
!regularization_losses

Elayers
Fnon_trainable_variables
Glayer_metrics
Hmetrics
"trainable_variables
[Y
VARIABLE_VALUEhidden_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1
 

$0
%1
≠
Ilayer_regularization_losses
'	variables
(regularization_losses

Jlayers
Knon_trainable_variables
Llayer_metrics
Mmetrics
)trainable_variables
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1
 

+0
,1
≠
Nlayer_regularization_losses
.	variables
/regularization_losses

Olayers
Pnon_trainable_variables
Qlayer_metrics
Rmetrics
0trainable_variables
KI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
3
4
5
 
 

S0
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
	Ttotal
	Ucount
V	variables
W	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

V	variables
ЦУ
VARIABLE_VALUE#Adagrad/hidden_0/kernel/accumulator\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE!Adagrad/hidden_0/bias/accumulatorZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#Adagrad/hidden_1/kernel/accumulator\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE!Adagrad/hidden_1/bias/accumulatorZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#Adagrad/hidden_2/kernel/accumulator\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE!Adagrad/hidden_2/bias/accumulatorZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE#Adagrad/hidden_3/kernel/accumulator\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE!Adagrad/hidden_3/bias/accumulatorZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE'Adagrad/output_layer/kernel/accumulator\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE%Adagrad/output_layer/bias/accumulatorZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_layerPlaceholder*'
_output_shapes
:€€€€€€€€€%*
dtype0*
shape:€€€€€€€€€%
ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerhidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *0
f+R)
'__inference_signature_wrapper_523492988
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
И
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_0/kernel/Read/ReadVariableOp!hidden_0/bias/Read/ReadVariableOp#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp#hidden_3/kernel/Read/ReadVariableOp!hidden_3/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adagrad/hidden_0/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_0/bias/accumulator/Read/ReadVariableOp7Adagrad/hidden_1/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_1/bias/accumulator/Read/ReadVariableOp7Adagrad/hidden_2/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_2/bias/accumulator/Read/ReadVariableOp7Adagrad/hidden_3/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_3/bias/accumulator/Read/ReadVariableOp;Adagrad/output_layer/kernel/accumulator/Read/ReadVariableOp9Adagrad/output_layer/bias/accumulator/Read/ReadVariableOpConst*&
Tin
2	*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_save_523493314
П
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biasoutput_layer/kerneloutput_layer/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotalcount#Adagrad/hidden_0/kernel/accumulator!Adagrad/hidden_0/bias/accumulator#Adagrad/hidden_1/kernel/accumulator!Adagrad/hidden_1/bias/accumulator#Adagrad/hidden_2/kernel/accumulator!Adagrad/hidden_2/bias/accumulator#Adagrad/hidden_3/kernel/accumulator!Adagrad/hidden_3/bias/accumulator'Adagrad/output_layer/kernel/accumulator%Adagrad/output_layer/bias/accumulator*%
Tin
2*
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
GPU2*0J 8В *.
f)R'
%__inference__traced_restore_523493399ай
ј
О
D__inference_model_layer_call_and_return_conditional_losses_523492934

inputs
hidden_0_523492908
hidden_0_523492910
hidden_1_523492913
hidden_1_523492915
hidden_2_523492918
hidden_2_523492920
hidden_3_523492923
hidden_3_523492925
output_layer_523492928
output_layer_523492930
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐ hidden_2/StatefulPartitionedCallҐ hidden_3/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCall°
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_523492908hidden_0_523492910*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_5234926942"
 hidden_0/StatefulPartitionedCallƒ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_523492913hidden_1_523492915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_5234927212"
 hidden_1/StatefulPartitionedCallƒ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_523492918hidden_2_523492920*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_5234927482"
 hidden_2/StatefulPartitionedCall√
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_523492923hidden_3_523492925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_3_layer_call_and_return_conditional_losses_5234927752"
 hidden_3/StatefulPartitionedCall„
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_523492928output_layer_523492930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_5234928022&
$output_layer/StatefulPartitionedCallі
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
Ґ7
ж
$__inference__wrapped_model_523492679
input_layer1
-model_hidden_0_matmul_readvariableop_resource2
.model_hidden_0_biasadd_readvariableop_resource1
-model_hidden_1_matmul_readvariableop_resource2
.model_hidden_1_biasadd_readvariableop_resource1
-model_hidden_2_matmul_readvariableop_resource2
.model_hidden_2_biasadd_readvariableop_resource1
-model_hidden_3_matmul_readvariableop_resource2
.model_hidden_3_biasadd_readvariableop_resource5
1model_output_layer_matmul_readvariableop_resource6
2model_output_layer_biasadd_readvariableop_resource
identityИҐ%model/hidden_0/BiasAdd/ReadVariableOpҐ$model/hidden_0/MatMul/ReadVariableOpҐ%model/hidden_1/BiasAdd/ReadVariableOpҐ$model/hidden_1/MatMul/ReadVariableOpҐ%model/hidden_2/BiasAdd/ReadVariableOpҐ$model/hidden_2/MatMul/ReadVariableOpҐ%model/hidden_3/BiasAdd/ReadVariableOpҐ$model/hidden_3/MatMul/ReadVariableOpҐ)model/output_layer/BiasAdd/ReadVariableOpҐ(model/output_layer/MatMul/ReadVariableOpї
$model/hidden_0/MatMul/ReadVariableOpReadVariableOp-model_hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%Р*
dtype02&
$model/hidden_0/MatMul/ReadVariableOp¶
model/hidden_0/MatMulMatMulinput_layer,model/hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
model/hidden_0/MatMulЇ
%model/hidden_0/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02'
%model/hidden_0/BiasAdd/ReadVariableOpЊ
model/hidden_0/BiasAddBiasAddmodel/hidden_0/MatMul:product:0-model/hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
model/hidden_0/BiasAddЖ
model/hidden_0/ReluRelumodel/hidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
model/hidden_0/ReluЉ
$model/hidden_1/MatMul/ReadVariableOpReadVariableOp-model_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
РР*
dtype02&
$model/hidden_1/MatMul/ReadVariableOpЉ
model/hidden_1/MatMulMatMul!model/hidden_0/Relu:activations:0,model/hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
model/hidden_1/MatMulЇ
%model/hidden_1/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02'
%model/hidden_1/BiasAdd/ReadVariableOpЊ
model/hidden_1/BiasAddBiasAddmodel/hidden_1/MatMul:product:0-model/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
model/hidden_1/BiasAddЖ
model/hidden_1/ReluRelumodel/hidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
model/hidden_1/ReluЉ
$model/hidden_2/MatMul/ReadVariableOpReadVariableOp-model_hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02&
$model/hidden_2/MatMul/ReadVariableOpЉ
model/hidden_2/MatMulMatMul!model/hidden_1/Relu:activations:0,model/hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
model/hidden_2/MatMulЇ
%model/hidden_2/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%model/hidden_2/BiasAdd/ReadVariableOpЊ
model/hidden_2/BiasAddBiasAddmodel/hidden_2/MatMul:product:0-model/hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
model/hidden_2/BiasAddЖ
model/hidden_2/ReluRelumodel/hidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
model/hidden_2/Reluї
$model/hidden_3/MatMul/ReadVariableOpReadVariableOp-model_hidden_3_matmul_readvariableop_resource*
_output_shapes
:	»d*
dtype02&
$model/hidden_3/MatMul/ReadVariableOpї
model/hidden_3/MatMulMatMul!model/hidden_2/Relu:activations:0,model/hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
model/hidden_3/MatMulє
%model/hidden_3/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02'
%model/hidden_3/BiasAdd/ReadVariableOpљ
model/hidden_3/BiasAddBiasAddmodel/hidden_3/MatMul:product:0-model/hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
model/hidden_3/BiasAddЕ
model/hidden_3/ReluRelumodel/hidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
model/hidden_3/Relu∆
(model/output_layer/MatMul/ReadVariableOpReadVariableOp1model_output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02*
(model/output_layer/MatMul/ReadVariableOp«
model/output_layer/MatMulMatMul!model/hidden_3/Relu:activations:00model/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
model/output_layer/MatMul≈
)model/output_layer/BiasAdd/ReadVariableOpReadVariableOp2model_output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02+
)model/output_layer/BiasAdd/ReadVariableOpЌ
model/output_layer/BiasAddBiasAdd#model/output_layer/MatMul:product:01model/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
model/output_layer/BiasAddЪ
model/output_layer/SoftmaxSoftmax#model/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
model/output_layer/SoftmaxЛ
IdentityIdentity$model/output_layer/Softmax:softmax:0&^model/hidden_0/BiasAdd/ReadVariableOp%^model/hidden_0/MatMul/ReadVariableOp&^model/hidden_1/BiasAdd/ReadVariableOp%^model/hidden_1/MatMul/ReadVariableOp&^model/hidden_2/BiasAdd/ReadVariableOp%^model/hidden_2/MatMul/ReadVariableOp&^model/hidden_3/BiasAdd/ReadVariableOp%^model/hidden_3/MatMul/ReadVariableOp*^model/output_layer/BiasAdd/ReadVariableOp)^model/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::2N
%model/hidden_0/BiasAdd/ReadVariableOp%model/hidden_0/BiasAdd/ReadVariableOp2L
$model/hidden_0/MatMul/ReadVariableOp$model/hidden_0/MatMul/ReadVariableOp2N
%model/hidden_1/BiasAdd/ReadVariableOp%model/hidden_1/BiasAdd/ReadVariableOp2L
$model/hidden_1/MatMul/ReadVariableOp$model/hidden_1/MatMul/ReadVariableOp2N
%model/hidden_2/BiasAdd/ReadVariableOp%model/hidden_2/BiasAdd/ReadVariableOp2L
$model/hidden_2/MatMul/ReadVariableOp$model/hidden_2/MatMul/ReadVariableOp2N
%model/hidden_3/BiasAdd/ReadVariableOp%model/hidden_3/BiasAdd/ReadVariableOp2L
$model/hidden_3/MatMul/ReadVariableOp$model/hidden_3/MatMul/ReadVariableOp2V
)model/output_layer/BiasAdd/ReadVariableOp)model/output_layer/BiasAdd/ReadVariableOp2T
(model/output_layer/MatMul/ReadVariableOp(model/output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
ѕ
У
D__inference_model_layer_call_and_return_conditional_losses_523492819
input_layer
hidden_0_523492705
hidden_0_523492707
hidden_1_523492732
hidden_1_523492734
hidden_2_523492759
hidden_2_523492761
hidden_3_523492786
hidden_3_523492788
output_layer_523492813
output_layer_523492815
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐ hidden_2/StatefulPartitionedCallҐ hidden_3/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCall¶
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_523492705hidden_0_523492707*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_5234926942"
 hidden_0/StatefulPartitionedCallƒ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_523492732hidden_1_523492734*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_5234927212"
 hidden_1/StatefulPartitionedCallƒ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_523492759hidden_2_523492761*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_5234927482"
 hidden_2/StatefulPartitionedCall√
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_523492786hidden_3_523492788*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_3_layer_call_and_return_conditional_losses_5234927752"
 hidden_3/StatefulPartitionedCall„
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_523492813output_layer_523492815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_5234928022&
$output_layer/StatefulPartitionedCallі
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
ф	
а
G__inference_hidden_3_layer_call_and_return_conditional_losses_523492775

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
ч	
а
G__inference_hidden_0_layer_call_and_return_conditional_losses_523493127

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
ъ	
а
G__inference_hidden_1_layer_call_and_return_conditional_losses_523492721

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
РР*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
ф	
а
G__inference_hidden_3_layer_call_and_return_conditional_losses_523493187

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
Е1
Й
D__inference_model_layer_call_and_return_conditional_losses_523493027

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource+
'hidden_1_matmul_readvariableop_resource,
(hidden_1_biasadd_readvariableop_resource+
'hidden_2_matmul_readvariableop_resource,
(hidden_2_biasadd_readvariableop_resource+
'hidden_3_matmul_readvariableop_resource,
(hidden_3_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИҐhidden_0/BiasAdd/ReadVariableOpҐhidden_0/MatMul/ReadVariableOpҐhidden_1/BiasAdd/ReadVariableOpҐhidden_1/MatMul/ReadVariableOpҐhidden_2/BiasAdd/ReadVariableOpҐhidden_2/MatMul/ReadVariableOpҐhidden_3/BiasAdd/ReadVariableOpҐhidden_3/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOp©
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%Р*
dtype02 
hidden_0/MatMul/ReadVariableOpП
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_0/MatMul®
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02!
hidden_0/BiasAdd/ReadVariableOp¶
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_0/Relu™
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
РР*
dtype02 
hidden_1/MatMul/ReadVariableOp§
hidden_1/MatMulMatMulhidden_0/Relu:activations:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_1/MatMul®
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02!
hidden_1/BiasAdd/ReadVariableOp¶
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_1/Relu™
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02 
hidden_2/MatMul/ReadVariableOp§
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_2/MatMul®
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02!
hidden_2/BiasAdd/ReadVariableOp¶
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_2/BiasAddt
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_2/Relu©
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes
:	»d*
dtype02 
hidden_3/MatMul/ReadVariableOp£
hidden_3/MatMulMatMulhidden_2/Relu:activations:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
hidden_3/MatMulІ
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
hidden_3/BiasAdd/ReadVariableOp•
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
hidden_3/BiasAdds
hidden_3/ReluReluhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
hidden_3/Reluі
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02$
"output_layer/MatMul/ReadVariableOpѓ
output_layer/MatMulMatMulhidden_3/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
output_layer/MatMul≥
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
output_layer/BiasAddИ
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
output_layer/Softmax…
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
ъ	
а
G__inference_hidden_2_layer_call_and_return_conditional_losses_523493167

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
ј
О
D__inference_model_layer_call_and_return_conditional_losses_523492880

inputs
hidden_0_523492854
hidden_0_523492856
hidden_1_523492859
hidden_1_523492861
hidden_2_523492864
hidden_2_523492866
hidden_3_523492869
hidden_3_523492871
output_layer_523492874
output_layer_523492876
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐ hidden_2/StatefulPartitionedCallҐ hidden_3/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCall°
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_523492854hidden_0_523492856*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_5234926942"
 hidden_0/StatefulPartitionedCallƒ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_523492859hidden_1_523492861*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_5234927212"
 hidden_1/StatefulPartitionedCallƒ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_523492864hidden_2_523492866*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_5234927482"
 hidden_2/StatefulPartitionedCall√
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_523492869hidden_3_523492871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_3_layer_call_and_return_conditional_losses_5234927752"
 hidden_3/StatefulPartitionedCall„
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_523492874output_layer_523492876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_5234928022&
$output_layer/StatefulPartitionedCallі
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
ы=
–
"__inference__traced_save_523493314
file_prefix.
*savev2_hidden_0_kernel_read_readvariableop,
(savev2_hidden_0_bias_read_readvariableop.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop.
*savev2_hidden_3_kernel_read_readvariableop,
(savev2_hidden_3_bias_read_readvariableop2
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
<savev2_adagrad_hidden_2_bias_accumulator_read_readvariableopB
>savev2_adagrad_hidden_3_kernel_accumulator_read_readvariableop@
<savev2_adagrad_hidden_3_bias_accumulator_read_readvariableopF
Bsavev2_adagrad_output_layer_kernel_accumulator_read_readvariableopD
@savev2_adagrad_output_layer_bias_accumulator_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameа
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*т
valueиBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЉ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices–
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_0_kernel_read_readvariableop(savev2_hidden_0_bias_read_readvariableop*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop*savev2_hidden_3_kernel_read_readvariableop(savev2_hidden_3_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adagrad_hidden_0_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_0_bias_accumulator_read_readvariableop>savev2_adagrad_hidden_1_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_1_bias_accumulator_read_readvariableop>savev2_adagrad_hidden_2_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_2_bias_accumulator_read_readvariableop>savev2_adagrad_hidden_3_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_3_bias_accumulator_read_readvariableopBsavev2_adagrad_output_layer_kernel_accumulator_read_readvariableop@savev2_adagrad_output_layer_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*’
_input_shapes√
ј: :	%Р:Р:
РР:Р:
Р»:»:	»d:d:d$:$: : : : : :	%Р:Р:
РР:Р:
Р»:»:	»d:d:d$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	%Р:!

_output_shapes	
:Р:&"
 
_output_shapes
:
РР:!

_output_shapes	
:Р:&"
 
_output_shapes
:
Р»:!

_output_shapes	
:»:%!

_output_shapes
:	»d: 

_output_shapes
:d:$	 

_output_shapes

:d$: 


_output_shapes
:$:
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
: :%!

_output_shapes
:	%Р:!

_output_shapes	
:Р:&"
 
_output_shapes
:
РР:!

_output_shapes	
:Р:&"
 
_output_shapes
:
Р»:!

_output_shapes	
:»:%!

_output_shapes
:	»d: 

_output_shapes
:d:$ 

_output_shapes

:d$: 

_output_shapes
:$:

_output_shapes
: 
а
ц
)__inference_model_layer_call_fn_523493116

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5234929342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
и
Б
,__inference_hidden_0_layer_call_fn_523493136

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_5234926942
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€%::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
и
Б
,__inference_hidden_3_layer_call_fn_523493196

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_3_layer_call_and_return_conditional_losses_5234927752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
о
Е
0__inference_output_layer_layer_call_fn_523493216

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_5234928022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ч	
а
G__inference_hidden_0_layer_call_and_return_conditional_losses_523492694

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
э	
д
K__inference_output_layer_layer_call_and_return_conditional_losses_523493207

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Пn
М
%__inference__traced_restore_523493399
file_prefix$
 assignvariableop_hidden_0_kernel$
 assignvariableop_1_hidden_0_bias&
"assignvariableop_2_hidden_1_kernel$
 assignvariableop_3_hidden_1_bias&
"assignvariableop_4_hidden_2_kernel$
 assignvariableop_5_hidden_2_bias&
"assignvariableop_6_hidden_3_kernel$
 assignvariableop_7_hidden_3_bias*
&assignvariableop_8_output_layer_kernel(
$assignvariableop_9_output_layer_bias$
 assignvariableop_10_adagrad_iter%
!assignvariableop_11_adagrad_decay-
)assignvariableop_12_adagrad_learning_rate
assignvariableop_13_total
assignvariableop_14_count;
7assignvariableop_15_adagrad_hidden_0_kernel_accumulator9
5assignvariableop_16_adagrad_hidden_0_bias_accumulator;
7assignvariableop_17_adagrad_hidden_1_kernel_accumulator9
5assignvariableop_18_adagrad_hidden_1_bias_accumulator;
7assignvariableop_19_adagrad_hidden_2_kernel_accumulator9
5assignvariableop_20_adagrad_hidden_2_bias_accumulator;
7assignvariableop_21_adagrad_hidden_3_kernel_accumulator9
5assignvariableop_22_adagrad_hidden_3_bias_accumulator?
;assignvariableop_23_adagrad_output_layer_kernel_accumulator=
9assignvariableop_24_adagrad_output_layer_bias_accumulator
identity_26ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*т
valueиBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices≠
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_hidden_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4І
AssignVariableOp_4AssignVariableOp"assignvariableop_4_hidden_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5•
AssignVariableOp_5AssignVariableOp assignvariableop_5_hidden_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp"assignvariableop_6_hidden_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_hidden_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ђ
AssignVariableOp_8AssignVariableOp&assignvariableop_8_output_layer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_output_layer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp assignvariableop_10_adagrad_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_adagrad_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adagrad_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13°
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14°
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15њ
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adagrad_hidden_0_kernel_accumulatorIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16љ
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adagrad_hidden_0_bias_accumulatorIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17њ
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adagrad_hidden_1_kernel_accumulatorIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18љ
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adagrad_hidden_1_bias_accumulatorIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19њ
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adagrad_hidden_2_kernel_accumulatorIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20љ
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adagrad_hidden_2_bias_accumulatorIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21њ
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adagrad_hidden_3_kernel_accumulatorIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22љ
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adagrad_hidden_3_bias_accumulatorIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23√
AssignVariableOp_23AssignVariableOp;assignvariableop_23_adagrad_output_layer_kernel_accumulatorIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ѕ
AssignVariableOp_24AssignVariableOp9assignvariableop_24_adagrad_output_layer_bias_accumulatorIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpД
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25ч
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
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
AssignVariableOp_24AssignVariableOp_242(
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
Е1
Й
D__inference_model_layer_call_and_return_conditional_losses_523493066

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource+
'hidden_1_matmul_readvariableop_resource,
(hidden_1_biasadd_readvariableop_resource+
'hidden_2_matmul_readvariableop_resource,
(hidden_2_biasadd_readvariableop_resource+
'hidden_3_matmul_readvariableop_resource,
(hidden_3_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИҐhidden_0/BiasAdd/ReadVariableOpҐhidden_0/MatMul/ReadVariableOpҐhidden_1/BiasAdd/ReadVariableOpҐhidden_1/MatMul/ReadVariableOpҐhidden_2/BiasAdd/ReadVariableOpҐhidden_2/MatMul/ReadVariableOpҐhidden_3/BiasAdd/ReadVariableOpҐhidden_3/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOp©
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%Р*
dtype02 
hidden_0/MatMul/ReadVariableOpП
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_0/MatMul®
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02!
hidden_0/BiasAdd/ReadVariableOp¶
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_0/Relu™
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
РР*
dtype02 
hidden_1/MatMul/ReadVariableOp§
hidden_1/MatMulMatMulhidden_0/Relu:activations:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_1/MatMul®
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02!
hidden_1/BiasAdd/ReadVariableOp¶
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
hidden_1/Relu™
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02 
hidden_2/MatMul/ReadVariableOp§
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_2/MatMul®
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02!
hidden_2/BiasAdd/ReadVariableOp¶
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_2/BiasAddt
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_2/Relu©
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes
:	»d*
dtype02 
hidden_3/MatMul/ReadVariableOp£
hidden_3/MatMulMatMulhidden_2/Relu:activations:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
hidden_3/MatMulІ
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
hidden_3/BiasAdd/ReadVariableOp•
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d2
hidden_3/BiasAdds
hidden_3/ReluReluhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
hidden_3/Reluі
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02$
"output_layer/MatMul/ReadVariableOpѓ
output_layer/MatMulMatMulhidden_3/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
output_layer/MatMul≥
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
output_layer/BiasAddИ
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2
output_layer/Softmax…
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
ъ	
а
G__inference_hidden_2_layer_call_and_return_conditional_losses_523492748

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Р»*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
э	
д
K__inference_output_layer_layer_call_and_return_conditional_losses_523492802

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
п
ы
)__inference_model_layer_call_fn_523492903
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5234928802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
ѕ
У
D__inference_model_layer_call_and_return_conditional_losses_523492848
input_layer
hidden_0_523492822
hidden_0_523492824
hidden_1_523492827
hidden_1_523492829
hidden_2_523492832
hidden_2_523492834
hidden_3_523492837
hidden_3_523492839
output_layer_523492842
output_layer_523492844
identityИҐ hidden_0/StatefulPartitionedCallҐ hidden_1/StatefulPartitionedCallҐ hidden_2/StatefulPartitionedCallҐ hidden_3/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCall¶
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_523492822hidden_0_523492824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_0_layer_call_and_return_conditional_losses_5234926942"
 hidden_0/StatefulPartitionedCallƒ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_523492827hidden_1_523492829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_5234927212"
 hidden_1/StatefulPartitionedCallƒ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_523492832hidden_2_523492834*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_5234927482"
 hidden_2/StatefulPartitionedCall√
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_523492837hidden_3_523492839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_3_layer_call_and_return_conditional_losses_5234927752"
 hidden_3/StatefulPartitionedCall„
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_523492842output_layer_523492844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_output_layer_layer_call_and_return_conditional_losses_5234928022&
$output_layer/StatefulPartitionedCallі
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
а
ц
)__inference_model_layer_call_fn_523493091

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5234928802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
п
ы
)__inference_model_layer_call_fn_523492957
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_5234929342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
к
Б
,__inference_hidden_2_layer_call_fn_523493176

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_2_layer_call_and_return_conditional_losses_5234927482
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
ъ	
а
G__inference_hidden_1_layer_call_and_return_conditional_losses_523493147

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
РР*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Р2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
к
Б
,__inference_hidden_1_layer_call_fn_523493156

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_hidden_1_layer_call_and_return_conditional_losses_5234927212
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€Р2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Р::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
Ќ
щ
'__inference_signature_wrapper_523492988
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityИҐStatefulPartitionedCallƒ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference__wrapped_model_5234926792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:€€€€€€€€€%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ј
serving_default£
C
input_layer4
serving_default_input_layer:0€€€€€€€€€%@
output_layer0
StatefulPartitionedCall:0€€€€€€€€€$tensorflow/serving/predict:Кƒ
 6
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer

signatures
#	_self_saveable_object_factories

	variables
regularization_losses
trainable_variables
	keras_api
b_default_save_signature
c__call__
*d&call_and_return_all_conditional_losses"К3
_tf_keras_networkо2{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_1", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_2", "inbound_nodes": [[["hidden_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_3", "inbound_nodes": [[["hidden_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_3", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 37]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_1", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_2", "inbound_nodes": [[["hidden_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_3", "inbound_nodes": [[["hidden_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_3", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "cross_entropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.014999999664723873, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
Ш
#_self_saveable_object_factories"р
_tf_keras_input_layer–{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
Ш

kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Dense", "name": "hidden_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}}
Ъ

kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"–
_tf_keras_layerґ{"class_name": "Dense", "name": "hidden_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400]}}
Ъ

kernel
bias
#_self_saveable_object_factories
 	variables
!regularization_losses
"trainable_variables
#	keras_api
i__call__
*j&call_and_return_all_conditional_losses"–
_tf_keras_layerґ{"class_name": "Dense", "name": "hidden_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400]}}
Ъ

$kernel
%bias
#&_self_saveable_object_factories
'	variables
(regularization_losses
)trainable_variables
*	keras_api
k__call__
*l&call_and_return_all_conditional_losses"–
_tf_keras_layerґ{"class_name": "Dense", "name": "hidden_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
§

+kernel
,bias
#-_self_saveable_object_factories
.	variables
/regularization_losses
0trainable_variables
1	keras_api
m__call__
*n&call_and_return_all_conditional_losses"Џ
_tf_keras_layerј{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
щ
2iter
	3decay
4learning_rateaccumulatorXaccumulatorYaccumulatorZaccumulator[accumulator\accumulator]$accumulator^%accumulator_+accumulator`,accumulatora"
	optimizer
,
oserving_default"
signature_map
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
$6
%7
+8
,9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
$6
%7
+8
,9"
trackable_list_wrapper
 
5layer_regularization_losses

	variables
regularization_losses

6layers
7non_trainable_variables
8layer_metrics
9metrics
trainable_variables
c__call__
b_default_save_signature
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
": 	%Р2hidden_0/kernel
:Р2hidden_0/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
:layer_regularization_losses
	variables
regularization_losses

;layers
<non_trainable_variables
=layer_metrics
>metrics
trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
#:!
РР2hidden_1/kernel
:Р2hidden_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
?layer_regularization_losses
	variables
regularization_losses

@layers
Anon_trainable_variables
Blayer_metrics
Cmetrics
trainable_variables
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
#:!
Р»2hidden_2/kernel
:»2hidden_2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
Dlayer_regularization_losses
 	variables
!regularization_losses

Elayers
Fnon_trainable_variables
Glayer_metrics
Hmetrics
"trainable_variables
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
": 	»d2hidden_3/kernel
:d2hidden_3/bias
 "
trackable_dict_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
≠
Ilayer_regularization_losses
'	variables
(regularization_losses

Jlayers
Knon_trainable_variables
Llayer_metrics
Mmetrics
)trainable_variables
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
%:#d$2output_layer/kernel
:$2output_layer/bias
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
≠
Nlayer_regularization_losses
.	variables
/regularization_losses

Olayers
Pnon_trainable_variables
Qlayer_metrics
Rmetrics
0trainable_variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
S0"
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
ї
	Ttotal
	Ucount
V	variables
W	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
T0
U1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
4:2	%Р2#Adagrad/hidden_0/kernel/accumulator
.:,Р2!Adagrad/hidden_0/bias/accumulator
5:3
РР2#Adagrad/hidden_1/kernel/accumulator
.:,Р2!Adagrad/hidden_1/bias/accumulator
5:3
Р»2#Adagrad/hidden_2/kernel/accumulator
.:,»2!Adagrad/hidden_2/bias/accumulator
4:2	»d2#Adagrad/hidden_3/kernel/accumulator
-:+d2!Adagrad/hidden_3/bias/accumulator
7:5d$2'Adagrad/output_layer/kernel/accumulator
1:/$2%Adagrad/output_layer/bias/accumulator
ж2г
$__inference__wrapped_model_523492679Ї
Л≤З
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
annotations™ **Ґ'
%К"
input_layer€€€€€€€€€%
т2п
)__inference_model_layer_call_fn_523493091
)__inference_model_layer_call_fn_523492957
)__inference_model_layer_call_fn_523493116
)__inference_model_layer_call_fn_523492903ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
ё2џ
D__inference_model_layer_call_and_return_conditional_losses_523493027
D__inference_model_layer_call_and_return_conditional_losses_523493066
D__inference_model_layer_call_and_return_conditional_losses_523492819
D__inference_model_layer_call_and_return_conditional_losses_523492848ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
÷2”
,__inference_hidden_0_layer_call_fn_523493136Ґ
Щ≤Х
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
annotations™ *
 
с2о
G__inference_hidden_0_layer_call_and_return_conditional_losses_523493127Ґ
Щ≤Х
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
annotations™ *
 
÷2”
,__inference_hidden_1_layer_call_fn_523493156Ґ
Щ≤Х
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
annotations™ *
 
с2о
G__inference_hidden_1_layer_call_and_return_conditional_losses_523493147Ґ
Щ≤Х
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
annotations™ *
 
÷2”
,__inference_hidden_2_layer_call_fn_523493176Ґ
Щ≤Х
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
annotations™ *
 
с2о
G__inference_hidden_2_layer_call_and_return_conditional_losses_523493167Ґ
Щ≤Х
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
annotations™ *
 
÷2”
,__inference_hidden_3_layer_call_fn_523493196Ґ
Щ≤Х
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
annotations™ *
 
с2о
G__inference_hidden_3_layer_call_and_return_conditional_losses_523493187Ґ
Щ≤Х
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
annotations™ *
 
Џ2„
0__inference_output_layer_layer_call_fn_523493216Ґ
Щ≤Х
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
annotations™ *
 
х2т
K__inference_output_layer_layer_call_and_return_conditional_losses_523493207Ґ
Щ≤Х
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
annotations™ *
 
“Bѕ
'__inference_signature_wrapper_523492988input_layer"Ф
Н≤Й
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
annotations™ *
 І
$__inference__wrapped_model_523492679
$%+,4Ґ1
*Ґ'
%К"
input_layer€€€€€€€€€%
™ ";™8
6
output_layer&К#
output_layer€€€€€€€€€$®
G__inference_hidden_0_layer_call_and_return_conditional_losses_523493127]/Ґ,
%Ґ"
 К
inputs€€€€€€€€€%
™ "&Ґ#
К
0€€€€€€€€€Р
Ъ А
,__inference_hidden_0_layer_call_fn_523493136P/Ґ,
%Ґ"
 К
inputs€€€€€€€€€%
™ "К€€€€€€€€€Р©
G__inference_hidden_1_layer_call_and_return_conditional_losses_523493147^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "&Ґ#
К
0€€€€€€€€€Р
Ъ Б
,__inference_hidden_1_layer_call_fn_523493156Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "К€€€€€€€€€Р©
G__inference_hidden_2_layer_call_and_return_conditional_losses_523493167^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "&Ґ#
К
0€€€€€€€€€»
Ъ Б
,__inference_hidden_2_layer_call_fn_523493176Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "К€€€€€€€€€»®
G__inference_hidden_3_layer_call_and_return_conditional_losses_523493187]$%0Ґ-
&Ґ#
!К
inputs€€€€€€€€€»
™ "%Ґ"
К
0€€€€€€€€€d
Ъ А
,__inference_hidden_3_layer_call_fn_523493196P$%0Ґ-
&Ґ#
!К
inputs€€€€€€€€€»
™ "К€€€€€€€€€dє
D__inference_model_layer_call_and_return_conditional_losses_523492819q
$%+,<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ є
D__inference_model_layer_call_and_return_conditional_losses_523492848q
$%+,<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ і
D__inference_model_layer_call_and_return_conditional_losses_523493027l
$%+,7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ і
D__inference_model_layer_call_and_return_conditional_losses_523493066l
$%+,7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ С
)__inference_model_layer_call_fn_523492903d
$%+,<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p

 
™ "К€€€€€€€€€$С
)__inference_model_layer_call_fn_523492957d
$%+,<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p 

 
™ "К€€€€€€€€€$М
)__inference_model_layer_call_fn_523493091_
$%+,7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p

 
™ "К€€€€€€€€€$М
)__inference_model_layer_call_fn_523493116_
$%+,7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p 

 
™ "К€€€€€€€€€$Ђ
K__inference_output_layer_layer_call_and_return_conditional_losses_523493207\+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Г
0__inference_output_layer_layer_call_fn_523493216O+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€$Ї
'__inference_signature_wrapper_523492988О
$%+,CҐ@
Ґ 
9™6
4
input_layer%К"
input_layer€€€€€€€€€%";™8
6
output_layer&К#
output_layer€€€€€€€€€$