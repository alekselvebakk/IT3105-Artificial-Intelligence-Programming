��
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
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
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
�
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�$*$
shared_nameoutput_layer/kernel
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	�$*
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
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
	variables
	trainable_variables

	keras_api
%
#_self_saveable_object_factories
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
 
 
 
 

0
1
2
3

0
1
2
3
�

layers
metrics
regularization_losses
layer_regularization_losses
layer_metrics
	variables
non_trainable_variables
	trainable_variables
 
[Y
VARIABLE_VALUEhidden_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
�

layers
 metrics
regularization_losses
!layer_regularization_losses
"layer_metrics
	variables
#non_trainable_variables
trainable_variables
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
�

$layers
%metrics
regularization_losses
&layer_regularization_losses
'layer_metrics
	variables
(non_trainable_variables
trainable_variables

0
1
2

)0
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
	*total
	+count
,	variables
-	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

,	variables
~
serving_default_input_layerPlaceholder*'
_output_shapes
:���������%*
dtype0*
shape:���������%
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerhidden_0/kernelhidden_0/biasoutput_layer/kerneloutput_layer/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference_signature_wrapper_691
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_0/kernel/Read/ReadVariableOp!hidden_0/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin

2*
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
GPU2*0J 8� *%
f R
__inference__traced_save_834
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_0/kernelhidden_0/biasoutput_layer/kerneloutput_layer/biastotalcount*
Tin
	2*
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
GPU2*0J 8� *(
f#R!
__inference__traced_restore_862͉
�
�
>__inference_model_layer_call_and_return_conditional_losses_607
input_layer
hidden_0_574
hidden_0_576
output_layer_601
output_layer_603
identity�� hidden_0/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_574hidden_0_576*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_hidden_0_layer_call_and_return_conditional_losses_5632"
 hidden_0/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_601output_layer_603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_output_layer_layer_call_and_return_conditional_losses_5902&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
!__inference_signature_wrapper_691
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__wrapped_model_5482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
>__inference_model_layer_call_and_return_conditional_losses_709

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��hidden_0/BiasAdd/ReadVariableOp�hidden_0/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
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
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMulhidden_0/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
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
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_638

inputs
hidden_0_627
hidden_0_629
output_layer_632
output_layer_634
identity�� hidden_0/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_627hidden_0_629*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_hidden_0_layer_call_and_return_conditional_losses_5632"
 hidden_0/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_632output_layer_634*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_output_layer_layer_call_and_return_conditional_losses_5902&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_621
input_layer
hidden_0_610
hidden_0_612
output_layer_615
output_layer_617
identity�� hidden_0/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_610hidden_0_612*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_hidden_0_layer_call_and_return_conditional_losses_5632"
 hidden_0/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_615output_layer_617*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_output_layer_layer_call_and_return_conditional_losses_5902&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
__inference__traced_save_834
file_prefix.
*savev2_hidden_0_kernel_read_readvariableop,
(savev2_hidden_0_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_0_kernel_read_readvariableop(savev2_hidden_0_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
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

identity_1Identity_1:output:0*>
_input_shapes-
+: :	%�:�:	�$:$: : : 2(
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
:�:%!

_output_shapes
:	�$: 

_output_shapes
:$:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
A__inference_hidden_0_layer_call_and_return_conditional_losses_764

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
�
�
#__inference_model_layer_call_fn_753

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_6652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
#__inference_model_layer_call_fn_676
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_6652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
__inference__wrapped_model_548
input_layer1
-model_hidden_0_matmul_readvariableop_resource2
.model_hidden_0_biasadd_readvariableop_resource5
1model_output_layer_matmul_readvariableop_resource6
2model_output_layer_biasadd_readvariableop_resource
identity��%model/hidden_0/BiasAdd/ReadVariableOp�$model/hidden_0/MatMul/ReadVariableOp�)model/output_layer/BiasAdd/ReadVariableOp�(model/output_layer/MatMul/ReadVariableOp�
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
(model/output_layer/MatMul/ReadVariableOpReadVariableOp1model_output_layer_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype02*
(model/output_layer/MatMul/ReadVariableOp�
model/output_layer/MatMulMatMul!model/hidden_0/Relu:activations:00model/output_layer/MatMul/ReadVariableOp:value:0*
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
model/output_layer/Softmax�
IdentityIdentity$model/output_layer/Softmax:softmax:0&^model/hidden_0/BiasAdd/ReadVariableOp%^model/hidden_0/MatMul/ReadVariableOp*^model/output_layer/BiasAdd/ReadVariableOp)^model/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::2N
%model/hidden_0/BiasAdd/ReadVariableOp%model/hidden_0/BiasAdd/ReadVariableOp2L
$model/hidden_0/MatMul/ReadVariableOp$model/hidden_0/MatMul/ReadVariableOp2V
)model/output_layer/BiasAdd/ReadVariableOp)model/output_layer/BiasAdd/ReadVariableOp2T
(model/output_layer/MatMul/ReadVariableOp(model/output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
�
>__inference_model_layer_call_and_return_conditional_losses_727

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��hidden_0/BiasAdd/ReadVariableOp�hidden_0/MatMul/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
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
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMulhidden_0/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
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
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
#__inference_model_layer_call_fn_740

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_6382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�

*__inference_output_layer_layer_call_fn_793

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
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_output_layer_layer_call_and_return_conditional_losses_5902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
>__inference_model_layer_call_and_return_conditional_losses_665

inputs
hidden_0_654
hidden_0_656
output_layer_659
output_layer_661
identity�� hidden_0/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_654hidden_0_656*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_hidden_0_layer_call_and_return_conditional_losses_5632"
 hidden_0/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_659output_layer_661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_output_layer_layer_call_and_return_conditional_losses_5902&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������%
 
_user_specified_nameinputs
�
�
__inference__traced_restore_862
file_prefix$
 assignvariableop_hidden_0_kernel$
 assignvariableop_1_hidden_0_bias*
&assignvariableop_2_output_layer_kernel(
$assignvariableop_3_output_layer_bias
assignvariableop_4_total
assignvariableop_5_count

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
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
AssignVariableOp_2AssignVariableOp&assignvariableop_2_output_layer_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_output_layer_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_totalIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_countIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
E__inference_output_layer_layer_call_and_return_conditional_losses_590

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
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
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_model_layer_call_fn_649
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_6382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������%::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������%
%
_user_specified_nameinput_layer
�
{
&__inference_hidden_0_layer_call_fn_773

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
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_hidden_0_layer_call_and_return_conditional_losses_5632
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
�	
�
E__inference_output_layer_layer_call_and_return_conditional_losses_784

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
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
A__inference_hidden_0_layer_call_and_return_conditional_losses_563

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
 
_user_specified_nameinputs"�L
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
StatefulPartitionedCall:0���������$tensorflow/serving/predict:�g
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
	variables
	trainable_variables

	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_default_save_signature"�
_tf_keras_network�{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 37]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "cross_entropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.0005000000237487257, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
�
#_self_saveable_object_factories"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
1__call__
*2&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}}
�

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
"
	optimizer
,
5serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�

layers
metrics
regularization_losses
layer_regularization_losses
layer_metrics
	variables
non_trainable_variables
	trainable_variables
.__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
": 	%�2hidden_0/kernel
:�2hidden_0/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

layers
 metrics
regularization_losses
!layer_regularization_losses
"layer_metrics
	variables
#non_trainable_variables
trainable_variables
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
&:$	�$2output_layer/kernel
:$2output_layer/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

$layers
%metrics
regularization_losses
&layer_regularization_losses
'layer_metrics
	variables
(non_trainable_variables
trainable_variables
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
'
)0"
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
�
	*total
	+count
,	variables
-	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
*0
+1"
trackable_list_wrapper
-
,	variables"
_generic_user_object
�2�
#__inference_model_layer_call_fn_676
#__inference_model_layer_call_fn_649
#__inference_model_layer_call_fn_740
#__inference_model_layer_call_fn_753�
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
�2�
>__inference_model_layer_call_and_return_conditional_losses_727
>__inference_model_layer_call_and_return_conditional_losses_607
>__inference_model_layer_call_and_return_conditional_losses_709
>__inference_model_layer_call_and_return_conditional_losses_621�
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
__inference__wrapped_model_548�
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
�2�
&__inference_hidden_0_layer_call_fn_773�
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
A__inference_hidden_0_layer_call_and_return_conditional_losses_764�
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
*__inference_output_layer_layer_call_fn_793�
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
E__inference_output_layer_layer_call_and_return_conditional_losses_784�
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
!__inference_signature_wrapper_691input_layer"�
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
__inference__wrapped_model_548y4�1
*�'
%�"
input_layer���������%
� ";�8
6
output_layer&�#
output_layer���������$�
A__inference_hidden_0_layer_call_and_return_conditional_losses_764]/�,
%�"
 �
inputs���������%
� "&�#
�
0����������
� z
&__inference_hidden_0_layer_call_fn_773P/�,
%�"
 �
inputs���������%
� "������������
>__inference_model_layer_call_and_return_conditional_losses_607k<�9
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
>__inference_model_layer_call_and_return_conditional_losses_621k<�9
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
>__inference_model_layer_call_and_return_conditional_losses_709f7�4
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
>__inference_model_layer_call_and_return_conditional_losses_727f7�4
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
#__inference_model_layer_call_fn_649^<�9
2�/
%�"
input_layer���������%
p

 
� "����������$�
#__inference_model_layer_call_fn_676^<�9
2�/
%�"
input_layer���������%
p 

 
� "����������$�
#__inference_model_layer_call_fn_740Y7�4
-�*
 �
inputs���������%
p

 
� "����������$�
#__inference_model_layer_call_fn_753Y7�4
-�*
 �
inputs���������%
p 

 
� "����������$�
E__inference_output_layer_layer_call_and_return_conditional_losses_784]0�-
&�#
!�
inputs����������
� "%�"
�
0���������$
� ~
*__inference_output_layer_layer_call_fn_793P0�-
&�#
!�
inputs����������
� "����������$�
!__inference_signature_wrapper_691�C�@
� 
9�6
4
input_layer%�"
input_layer���������%";�8
6
output_layer&�#
output_layer���������$