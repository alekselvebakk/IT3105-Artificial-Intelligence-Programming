НЁ
щО
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
О
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8сд
{
hidden_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%* 
shared_namehidden_0/kernel
t
#hidden_0/kernel/Read/ReadVariableOpReadVariableOphidden_0/kernel*
_output_shapes
:	%*
dtype0
s
hidden_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_0/bias
l
!hidden_0/bias/Read/ReadVariableOpReadVariableOphidden_0/bias*
_output_shapes	
:*
dtype0
|
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namehidden_1/kernel
u
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel* 
_output_shapes
:
*
dtype0
s
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_1/bias
l
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes	
:*
dtype0
|
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ш* 
shared_namehidden_2/kernel
u
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel* 
_output_shapes
:
Ш*
dtype0
s
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*
shared_namehidden_2/bias
l
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes	
:Ш*
dtype0
{
hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Шd* 
shared_namehidden_3/kernel
t
#hidden_3/kernel/Read/ReadVariableOpReadVariableOphidden_3/kernel*
_output_shapes
:	Шd*
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

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

NoOpNoOp
Ѓ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*о
valueдBб BЪ
ц
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

regularization_losses
	variables
trainable_variables
	keras_api
%
#_self_saveable_object_factories


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


kernel
bias
#_self_saveable_object_factories
 regularization_losses
!	variables
"trainable_variables
#	keras_api


$kernel
%bias
#&_self_saveable_object_factories
'regularization_losses
(	variables
)trainable_variables
*	keras_api


+kernel
,bias
#-_self_saveable_object_factories
.regularization_losses
/	variables
0trainable_variables
1	keras_api
 
 
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
­

regularization_losses
2layer_regularization_losses
3non_trainable_variables
4layer_metrics

5layers
	variables
trainable_variables
6metrics
 
[Y
VARIABLE_VALUEhidden_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
­
regularization_losses
7layer_regularization_losses
8non_trainable_variables
9layer_metrics

:layers
	variables
trainable_variables
;metrics
[Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
­
regularization_losses
<layer_regularization_losses
=non_trainable_variables
>layer_metrics

?layers
	variables
trainable_variables
@metrics
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
­
 regularization_losses
Alayer_regularization_losses
Bnon_trainable_variables
Clayer_metrics

Dlayers
!	variables
"trainable_variables
Emetrics
[Y
VARIABLE_VALUEhidden_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

$0
%1

$0
%1
­
'regularization_losses
Flayer_regularization_losses
Gnon_trainable_variables
Hlayer_metrics

Ilayers
(	variables
)trainable_variables
Jmetrics
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

+0
,1

+0
,1
­
.regularization_losses
Klayer_regularization_losses
Lnon_trainable_variables
Mlayer_metrics

Nlayers
/	variables
0trainable_variables
Ometrics
 
 
 
*
0
1
2
3
4
5

P0
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
	Qtotal
	Rcount
S	variables
T	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1

S	variables
~
serving_default_input_layerPlaceholder*'
_output_shapes
:џџџџџџџџџ%*
dtype0*
shape:џџџџџџџџџ%
ѓ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerhidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_4514
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ю
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_0/kernel/Read/ReadVariableOp!hidden_0/bias/Read/ReadVariableOp#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp#hidden_3/kernel/Read/ReadVariableOp!hidden_3/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8 *&
f!R
__inference__traced_save_4801
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_0/kernelhidden_0/biashidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biasoutput_layer/kerneloutput_layer/biastotalcount*
Tin
2*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_restore_4847я
ф

+__inference_output_layer_layer_call_fn_4742

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_43322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
х
і
$__inference_model_layer_call_fn_4433
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
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_44102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:џџџџџџџџџ%
%
_user_specified_nameinput_layer
ѕ	
л
B__inference_hidden_1_layer_call_and_return_conditional_losses_4251

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х
і
$__inference_model_layer_call_fn_4487
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
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_44642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:џџџџџџџџџ%
%
_user_specified_nameinput_layer
1

?__inference_model_layer_call_and_return_conditional_losses_4592

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
identityЂhidden_0/BiasAdd/ReadVariableOpЂhidden_0/MatMul/ReadVariableOpЂhidden_1/BiasAdd/ReadVariableOpЂhidden_1/MatMul/ReadVariableOpЂhidden_2/BiasAdd/ReadVariableOpЂhidden_2/MatMul/ReadVariableOpЂhidden_3/BiasAdd/ReadVariableOpЂhidden_3/MatMul/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpЉ
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%*
dtype02 
hidden_0/MatMul/ReadVariableOp
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_0/MatMulЈ
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_0/BiasAdd/ReadVariableOpІ
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_0/ReluЊ
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
hidden_1/MatMul/ReadVariableOpЄ
hidden_1/MatMulMatMulhidden_0/Relu:activations:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_1/MatMulЈ
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_1/BiasAdd/ReadVariableOpІ
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_1/ReluЊ
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
Ш*
dtype02 
hidden_2/MatMul/ReadVariableOpЄ
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
hidden_2/MatMulЈ
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02!
hidden_2/BiasAdd/ReadVariableOpІ
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
hidden_2/BiasAddt
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
hidden_2/ReluЉ
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02 
hidden_3/MatMul/ReadVariableOpЃ
hidden_3/MatMulMatMulhidden_2/Relu:activations:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
hidden_3/MatMulЇ
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
hidden_3/BiasAdd/ReadVariableOpЅ
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
hidden_3/BiasAdds
hidden_3/ReluReluhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
hidden_3/ReluД
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02$
"output_layer/MatMul/ReadVariableOpЏ
output_layer/MatMulMatMulhidden_3/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
output_layer/MatMulГ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#output_layer/BiasAdd/ReadVariableOpЕ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
output_layer/BiasAdd
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
output_layer/SoftmaxЩ
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::2B
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
:џџџџџџџџџ%
 
_user_specified_nameinputs
У
є
"__inference_signature_wrapper_4514
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
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_42092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:џџџџџџџџџ%
%
_user_specified_nameinput_layer
О
з
?__inference_model_layer_call_and_return_conditional_losses_4410

inputs
hidden_0_4384
hidden_0_4386
hidden_1_4389
hidden_1_4391
hidden_2_4394
hidden_2_4396
hidden_3_4399
hidden_3_4401
output_layer_4404
output_layer_4406
identityЂ hidden_0/StatefulPartitionedCallЂ hidden_1/StatefulPartitionedCallЂ hidden_2/StatefulPartitionedCallЂ hidden_3/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCall
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_4384hidden_0_4386*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_0_layer_call_and_return_conditional_losses_42242"
 hidden_0/StatefulPartitionedCallЕ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_4389hidden_1_4391*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_1_layer_call_and_return_conditional_losses_42512"
 hidden_1/StatefulPartitionedCallЕ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_4394hidden_2_4396*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_2_layer_call_and_return_conditional_losses_42782"
 hidden_2/StatefulPartitionedCallД
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_4399hidden_3_4401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_3_layer_call_and_return_conditional_losses_43052"
 hidden_3/StatefulPartitionedCallШ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_4404output_layer_4406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_43322&
$output_layer/StatefulPartitionedCallД
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
Э
м
?__inference_model_layer_call_and_return_conditional_losses_4349
input_layer
hidden_0_4235
hidden_0_4237
hidden_1_4262
hidden_1_4264
hidden_2_4289
hidden_2_4291
hidden_3_4316
hidden_3_4318
output_layer_4343
output_layer_4345
identityЂ hidden_0/StatefulPartitionedCallЂ hidden_1/StatefulPartitionedCallЂ hidden_2/StatefulPartitionedCallЂ hidden_3/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCall
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_4235hidden_0_4237*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_0_layer_call_and_return_conditional_losses_42242"
 hidden_0/StatefulPartitionedCallЕ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_4262hidden_1_4264*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_1_layer_call_and_return_conditional_losses_42512"
 hidden_1/StatefulPartitionedCallЕ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_4289hidden_2_4291*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_2_layer_call_and_return_conditional_losses_42782"
 hidden_2/StatefulPartitionedCallД
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_4316hidden_3_4318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_3_layer_call_and_return_conditional_losses_43052"
 hidden_3/StatefulPartitionedCallШ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_4343output_layer_4345*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_43322&
$output_layer/StatefulPartitionedCallД
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:џџџџџџџџџ%
%
_user_specified_nameinput_layer
ѕ	
л
B__inference_hidden_1_layer_call_and_return_conditional_losses_4673

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л$

__inference__traced_save_4801
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
,savev2_output_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameГ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Х
valueЛBИB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesР
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_0_kernel_read_readvariableop(savev2_hidden_0_bias_read_readvariableop*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop*savev2_hidden_3_kernel_read_readvariableop(savev2_hidden_3_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*t
_input_shapesc
a: :	%::
::
Ш:Ш:	Шd:d:d$:$: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	%:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
Ш:!

_output_shapes	
:Ш:%!

_output_shapes
:	Шd: 
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
: 
5

 __inference__traced_restore_4847
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
$assignvariableop_9_output_layer_bias
assignvariableop_10_total
assignvariableop_11_count
identity_13ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Й
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Х
valueЛBИB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_hidden_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_hidden_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_hidden_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ї
AssignVariableOp_6AssignVariableOp"assignvariableop_6_hidden_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѕ
AssignVariableOp_7AssignVariableOp assignvariableop_7_hidden_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ћ
AssignVariableOp_8AssignVariableOp&assignvariableop_8_output_layer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Љ
AssignVariableOp_9AssignVariableOp$assignvariableop_9_output_layer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ё
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ё
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpц
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12й
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
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
ж
ё
$__inference_model_layer_call_fn_4642

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
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_44642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
7
с
__inference__wrapped_model_4209
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
identityЂ%model/hidden_0/BiasAdd/ReadVariableOpЂ$model/hidden_0/MatMul/ReadVariableOpЂ%model/hidden_1/BiasAdd/ReadVariableOpЂ$model/hidden_1/MatMul/ReadVariableOpЂ%model/hidden_2/BiasAdd/ReadVariableOpЂ$model/hidden_2/MatMul/ReadVariableOpЂ%model/hidden_3/BiasAdd/ReadVariableOpЂ$model/hidden_3/MatMul/ReadVariableOpЂ)model/output_layer/BiasAdd/ReadVariableOpЂ(model/output_layer/MatMul/ReadVariableOpЛ
$model/hidden_0/MatMul/ReadVariableOpReadVariableOp-model_hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%*
dtype02&
$model/hidden_0/MatMul/ReadVariableOpІ
model/hidden_0/MatMulMatMulinput_layer,model/hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/hidden_0/MatMulК
%model/hidden_0/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%model/hidden_0/BiasAdd/ReadVariableOpО
model/hidden_0/BiasAddBiasAddmodel/hidden_0/MatMul:product:0-model/hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/hidden_0/BiasAdd
model/hidden_0/ReluRelumodel/hidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/hidden_0/ReluМ
$model/hidden_1/MatMul/ReadVariableOpReadVariableOp-model_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$model/hidden_1/MatMul/ReadVariableOpМ
model/hidden_1/MatMulMatMul!model/hidden_0/Relu:activations:0,model/hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/hidden_1/MatMulК
%model/hidden_1/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%model/hidden_1/BiasAdd/ReadVariableOpО
model/hidden_1/BiasAddBiasAddmodel/hidden_1/MatMul:product:0-model/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/hidden_1/BiasAdd
model/hidden_1/ReluRelumodel/hidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/hidden_1/ReluМ
$model/hidden_2/MatMul/ReadVariableOpReadVariableOp-model_hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
Ш*
dtype02&
$model/hidden_2/MatMul/ReadVariableOpМ
model/hidden_2/MatMulMatMul!model/hidden_1/Relu:activations:0,model/hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
model/hidden_2/MatMulК
%model/hidden_2/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02'
%model/hidden_2/BiasAdd/ReadVariableOpО
model/hidden_2/BiasAddBiasAddmodel/hidden_2/MatMul:product:0-model/hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
model/hidden_2/BiasAdd
model/hidden_2/ReluRelumodel/hidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
model/hidden_2/ReluЛ
$model/hidden_3/MatMul/ReadVariableOpReadVariableOp-model_hidden_3_matmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02&
$model/hidden_3/MatMul/ReadVariableOpЛ
model/hidden_3/MatMulMatMul!model/hidden_2/Relu:activations:0,model/hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
model/hidden_3/MatMulЙ
%model/hidden_3/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02'
%model/hidden_3/BiasAdd/ReadVariableOpН
model/hidden_3/BiasAddBiasAddmodel/hidden_3/MatMul:product:0-model/hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
model/hidden_3/BiasAdd
model/hidden_3/ReluRelumodel/hidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
model/hidden_3/ReluЦ
(model/output_layer/MatMul/ReadVariableOpReadVariableOp1model_output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02*
(model/output_layer/MatMul/ReadVariableOpЧ
model/output_layer/MatMulMatMul!model/hidden_3/Relu:activations:00model/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
model/output_layer/MatMulХ
)model/output_layer/BiasAdd/ReadVariableOpReadVariableOp2model_output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02+
)model/output_layer/BiasAdd/ReadVariableOpЭ
model/output_layer/BiasAddBiasAdd#model/output_layer/MatMul:product:01model/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
model/output_layer/BiasAdd
model/output_layer/SoftmaxSoftmax#model/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
model/output_layer/Softmax
IdentityIdentity$model/output_layer/Softmax:softmax:0&^model/hidden_0/BiasAdd/ReadVariableOp%^model/hidden_0/MatMul/ReadVariableOp&^model/hidden_1/BiasAdd/ReadVariableOp%^model/hidden_1/MatMul/ReadVariableOp&^model/hidden_2/BiasAdd/ReadVariableOp%^model/hidden_2/MatMul/ReadVariableOp&^model/hidden_3/BiasAdd/ReadVariableOp%^model/hidden_3/MatMul/ReadVariableOp*^model/output_layer/BiasAdd/ReadVariableOp)^model/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::2N
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
:џџџџџџџџџ%
%
_user_specified_nameinput_layer
н
|
'__inference_hidden_0_layer_call_fn_4662

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_0_layer_call_and_return_conditional_losses_42242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ%::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
ђ	
л
B__inference_hidden_0_layer_call_and_return_conditional_losses_4653

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
н
|
'__inference_hidden_3_layer_call_fn_4722

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_3_layer_call_and_return_conditional_losses_43052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџШ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџШ
 
_user_specified_nameinputs
ђ	
л
B__inference_hidden_0_layer_call_and_return_conditional_losses_4224

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
я	
л
B__inference_hidden_3_layer_call_and_return_conditional_losses_4305

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџШ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџШ
 
_user_specified_nameinputs
ј	
п
F__inference_output_layer_layer_call_and_return_conditional_losses_4733

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ$2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
О
з
?__inference_model_layer_call_and_return_conditional_losses_4464

inputs
hidden_0_4438
hidden_0_4440
hidden_1_4443
hidden_1_4445
hidden_2_4448
hidden_2_4450
hidden_3_4453
hidden_3_4455
output_layer_4458
output_layer_4460
identityЂ hidden_0/StatefulPartitionedCallЂ hidden_1/StatefulPartitionedCallЂ hidden_2/StatefulPartitionedCallЂ hidden_3/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCall
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_4438hidden_0_4440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_0_layer_call_and_return_conditional_losses_42242"
 hidden_0/StatefulPartitionedCallЕ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_4443hidden_1_4445*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_1_layer_call_and_return_conditional_losses_42512"
 hidden_1/StatefulPartitionedCallЕ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_4448hidden_2_4450*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_2_layer_call_and_return_conditional_losses_42782"
 hidden_2/StatefulPartitionedCallД
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_4453hidden_3_4455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_3_layer_call_and_return_conditional_losses_43052"
 hidden_3/StatefulPartitionedCallШ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_4458output_layer_4460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_43322&
$output_layer/StatefulPartitionedCallД
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
я	
л
B__inference_hidden_3_layer_call_and_return_conditional_losses_4713

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџШ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџШ
 
_user_specified_nameinputs
ѕ	
л
B__inference_hidden_2_layer_call_and_return_conditional_losses_4693

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџШ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п
|
'__inference_hidden_2_layer_call_fn_4702

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_2_layer_call_and_return_conditional_losses_42782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџШ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѕ	
л
B__inference_hidden_2_layer_call_and_return_conditional_losses_4278

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџШ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј	
п
F__inference_output_layer_layer_call_and_return_conditional_losses_4332

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ$2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
п
|
'__inference_hidden_1_layer_call_fn_4682

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_1_layer_call_and_return_conditional_losses_42512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
ё
$__inference_model_layer_call_fn_4617

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
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_44102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
1

?__inference_model_layer_call_and_return_conditional_losses_4553

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
identityЂhidden_0/BiasAdd/ReadVariableOpЂhidden_0/MatMul/ReadVariableOpЂhidden_1/BiasAdd/ReadVariableOpЂhidden_1/MatMul/ReadVariableOpЂhidden_2/BiasAdd/ReadVariableOpЂhidden_2/MatMul/ReadVariableOpЂhidden_3/BiasAdd/ReadVariableOpЂhidden_3/MatMul/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpЉ
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%*
dtype02 
hidden_0/MatMul/ReadVariableOp
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_0/MatMulЈ
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_0/BiasAdd/ReadVariableOpІ
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_0/ReluЊ
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
hidden_1/MatMul/ReadVariableOpЄ
hidden_1/MatMulMatMulhidden_0/Relu:activations:0&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_1/MatMulЈ
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_1/BiasAdd/ReadVariableOpІ
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
hidden_1/ReluЊ
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
Ш*
dtype02 
hidden_2/MatMul/ReadVariableOpЄ
hidden_2/MatMulMatMulhidden_1/Relu:activations:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
hidden_2/MatMulЈ
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02!
hidden_2/BiasAdd/ReadVariableOpІ
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
hidden_2/BiasAddt
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџШ2
hidden_2/ReluЉ
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes
:	Шd*
dtype02 
hidden_3/MatMul/ReadVariableOpЃ
hidden_3/MatMulMatMulhidden_2/Relu:activations:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
hidden_3/MatMulЇ
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
hidden_3/BiasAdd/ReadVariableOpЅ
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
hidden_3/BiasAdds
hidden_3/ReluReluhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
hidden_3/ReluД
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d$*
dtype02$
"output_layer/MatMul/ReadVariableOpЏ
output_layer/MatMulMatMulhidden_3/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
output_layer/MatMulГ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02%
#output_layer/BiasAdd/ReadVariableOpЕ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
output_layer/BiasAdd
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ$2
output_layer/SoftmaxЩ
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::2B
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
:џџџџџџџџџ%
 
_user_specified_nameinputs
Э
м
?__inference_model_layer_call_and_return_conditional_losses_4378
input_layer
hidden_0_4352
hidden_0_4354
hidden_1_4357
hidden_1_4359
hidden_2_4362
hidden_2_4364
hidden_3_4367
hidden_3_4369
output_layer_4372
output_layer_4374
identityЂ hidden_0/StatefulPartitionedCallЂ hidden_1/StatefulPartitionedCallЂ hidden_2/StatefulPartitionedCallЂ hidden_3/StatefulPartitionedCallЂ$output_layer/StatefulPartitionedCall
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_4352hidden_0_4354*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_0_layer_call_and_return_conditional_losses_42242"
 hidden_0/StatefulPartitionedCallЕ
 hidden_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0hidden_1_4357hidden_1_4359*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_1_layer_call_and_return_conditional_losses_42512"
 hidden_1/StatefulPartitionedCallЕ
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_4362hidden_2_4364*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџШ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_2_layer_call_and_return_conditional_losses_42782"
 hidden_2/StatefulPartitionedCallД
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_4367hidden_3_4369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_hidden_3_layer_call_and_return_conditional_losses_43052"
 hidden_3/StatefulPartitionedCallШ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0output_layer_4372output_layer_4374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_43322&
$output_layer/StatefulPartitionedCallД
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ$2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ%::::::::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:џџџџџџџџџ%
%
_user_specified_nameinput_layer"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*З
serving_defaultЃ
C
input_layer4
serving_default_input_layer:0џџџџџџџџџ%@
output_layer0
StatefulPartitionedCall:0џџџџџџџџџ$tensorflow/serving/predict:џЛ
Ъ6
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

regularization_losses
	variables
trainable_variables
	keras_api
*U&call_and_return_all_conditional_losses
V_default_save_signature
W__call__"3
_tf_keras_networkю2{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_1", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_2", "inbound_nodes": [[["hidden_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_3", "inbound_nodes": [[["hidden_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_3", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 37]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_1", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_2", "inbound_nodes": [[["hidden_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_3", "inbound_nodes": [[["hidden_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_3", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "cross_entropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.014999999664723873, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}

#_self_saveable_object_factories"№
_tf_keras_input_layerа{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"Ю
_tf_keras_layerД{"class_name": "Dense", "name": "hidden_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}}


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "hidden_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400]}}


kernel
bias
#_self_saveable_object_factories
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*\&call_and_return_all_conditional_losses
]__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "hidden_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400]}}


$kernel
%bias
#&_self_saveable_object_factories
'regularization_losses
(	variables
)trainable_variables
*	keras_api
*^&call_and_return_all_conditional_losses
___call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "hidden_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
Є

+kernel
,bias
#-_self_saveable_object_factories
.regularization_losses
/	variables
0trainable_variables
1	keras_api
*`&call_and_return_all_conditional_losses
a__call__"к
_tf_keras_layerР{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
"
	optimizer
,
bserving_default"
signature_map
 "
trackable_dict_wrapper
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
Ъ

regularization_losses
2layer_regularization_losses
3non_trainable_variables
4layer_metrics

5layers
	variables
trainable_variables
6metrics
W__call__
V_default_save_signature
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
": 	%2hidden_0/kernel
:2hidden_0/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
7layer_regularization_losses
8non_trainable_variables
9layer_metrics

:layers
	variables
trainable_variables
;metrics
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
#:!
2hidden_1/kernel
:2hidden_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
<layer_regularization_losses
=non_trainable_variables
>layer_metrics

?layers
	variables
trainable_variables
@metrics
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
#:!
Ш2hidden_2/kernel
:Ш2hidden_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
 regularization_losses
Alayer_regularization_losses
Bnon_trainable_variables
Clayer_metrics

Dlayers
!	variables
"trainable_variables
Emetrics
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
": 	Шd2hidden_3/kernel
:d2hidden_3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
­
'regularization_losses
Flayer_regularization_losses
Gnon_trainable_variables
Hlayer_metrics

Ilayers
(	variables
)trainable_variables
Jmetrics
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
%:#d$2output_layer/kernel
:$2output_layer/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
­
.regularization_losses
Klayer_regularization_losses
Lnon_trainable_variables
Mlayer_metrics

Nlayers
/	variables
0trainable_variables
Ometrics
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
P0"
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
Л
	Qtotal
	Rcount
S	variables
T	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
Q0
R1"
trackable_list_wrapper
-
S	variables"
_generic_user_object
Ъ2Ч
?__inference_model_layer_call_and_return_conditional_losses_4553
?__inference_model_layer_call_and_return_conditional_losses_4349
?__inference_model_layer_call_and_return_conditional_losses_4592
?__inference_model_layer_call_and_return_conditional_losses_4378Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
с2о
__inference__wrapped_model_4209К
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_layerџџџџџџџџџ%
о2л
$__inference_model_layer_call_fn_4487
$__inference_model_layer_call_fn_4617
$__inference_model_layer_call_fn_4433
$__inference_model_layer_call_fn_4642Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_hidden_0_layer_call_and_return_conditional_losses_4653Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_hidden_0_layer_call_fn_4662Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_hidden_1_layer_call_and_return_conditional_losses_4673Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_hidden_1_layer_call_fn_4682Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_hidden_2_layer_call_and_return_conditional_losses_4693Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_hidden_2_layer_call_fn_4702Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_hidden_3_layer_call_and_return_conditional_losses_4713Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_hidden_3_layer_call_fn_4722Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_output_layer_layer_call_and_return_conditional_losses_4733Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_output_layer_layer_call_fn_4742Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЭBЪ
"__inference_signature_wrapper_4514input_layer"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Ђ
__inference__wrapped_model_4209
$%+,4Ђ1
*Ђ'
%"
input_layerџџџџџџџџџ%
Њ ";Њ8
6
output_layer&#
output_layerџџџџџџџџџ$Ѓ
B__inference_hidden_0_layer_call_and_return_conditional_losses_4653]/Ђ,
%Ђ"
 
inputsџџџџџџџџџ%
Њ "&Ђ#

0џџџџџџџџџ
 {
'__inference_hidden_0_layer_call_fn_4662P/Ђ,
%Ђ"
 
inputsџџџџџџџџџ%
Њ "џџџџџџџџџЄ
B__inference_hidden_1_layer_call_and_return_conditional_losses_4673^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 |
'__inference_hidden_1_layer_call_fn_4682Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЄ
B__inference_hidden_2_layer_call_and_return_conditional_losses_4693^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџШ
 |
'__inference_hidden_2_layer_call_fn_4702Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџШЃ
B__inference_hidden_3_layer_call_and_return_conditional_losses_4713]$%0Ђ-
&Ђ#
!
inputsџџџџџџџџџШ
Њ "%Ђ"

0џџџџџџџџџd
 {
'__inference_hidden_3_layer_call_fn_4722P$%0Ђ-
&Ђ#
!
inputsџџџџџџџџџШ
Њ "џџџџџџџџџdД
?__inference_model_layer_call_and_return_conditional_losses_4349q
$%+,<Ђ9
2Ђ/
%"
input_layerџџџџџџџџџ%
p

 
Њ "%Ђ"

0џџџџџџџџџ$
 Д
?__inference_model_layer_call_and_return_conditional_losses_4378q
$%+,<Ђ9
2Ђ/
%"
input_layerџџџџџџџџџ%
p 

 
Њ "%Ђ"

0џџџџџџџџџ$
 Џ
?__inference_model_layer_call_and_return_conditional_losses_4553l
$%+,7Ђ4
-Ђ*
 
inputsџџџџџџџџџ%
p

 
Њ "%Ђ"

0џџџџџџџџџ$
 Џ
?__inference_model_layer_call_and_return_conditional_losses_4592l
$%+,7Ђ4
-Ђ*
 
inputsџџџџџџџџџ%
p 

 
Њ "%Ђ"

0џџџџџџџџџ$
 
$__inference_model_layer_call_fn_4433d
$%+,<Ђ9
2Ђ/
%"
input_layerџџџџџџџџџ%
p

 
Њ "џџџџџџџџџ$
$__inference_model_layer_call_fn_4487d
$%+,<Ђ9
2Ђ/
%"
input_layerџџџџџџџџџ%
p 

 
Њ "џџџџџџџџџ$
$__inference_model_layer_call_fn_4617_
$%+,7Ђ4
-Ђ*
 
inputsџџџџџџџџџ%
p

 
Њ "џџџџџџџџџ$
$__inference_model_layer_call_fn_4642_
$%+,7Ђ4
-Ђ*
 
inputsџџџџџџџџџ%
p 

 
Њ "џџџџџџџџџ$І
F__inference_output_layer_layer_call_and_return_conditional_losses_4733\+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ$
 ~
+__inference_output_layer_layer_call_fn_4742O+,/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџ$Е
"__inference_signature_wrapper_4514
$%+,CЂ@
Ђ 
9Њ6
4
input_layer%"
input_layerџџџџџџџџџ%";Њ8
6
output_layer&#
output_layerџџџџџџџџџ$