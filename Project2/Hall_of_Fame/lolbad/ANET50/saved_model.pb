уф
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
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8уч
{
hidden_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	%»* 
shared_namehidden_0/kernel
t
#hidden_0/kernel/Read/ReadVariableOpReadVariableOphidden_0/kernel*
_output_shapes
:	%»*
dtype0
s
hidden_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*
shared_namehidden_0/bias
l
!hidden_0/bias/Read/ReadVariableOpReadVariableOphidden_0/bias*
_output_shapes	
:»*
dtype0
Г
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»$*$
shared_nameoutput_layer/kernel
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	»$*
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
shape:	%»*4
shared_name%#Adagrad/hidden_0/kernel/accumulator
Ь
7Adagrad/hidden_0/kernel/accumulator/Read/ReadVariableOpReadVariableOp#Adagrad/hidden_0/kernel/accumulator*
_output_shapes
:	%»*
dtype0
Ы
!Adagrad/hidden_0/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*2
shared_name#!Adagrad/hidden_0/bias/accumulator
Ф
5Adagrad/hidden_0/bias/accumulator/Read/ReadVariableOpReadVariableOp!Adagrad/hidden_0/bias/accumulator*
_output_shapes	
:»*
dtype0
Ђ
'Adagrad/output_layer/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»$*8
shared_name)'Adagrad/output_layer/kernel/accumulator
§
;Adagrad/output_layer/kernel/accumulator/Read/ReadVariableOpReadVariableOp'Adagrad/output_layer/kernel/accumulator*
_output_shapes
:	»$*
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
ђ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*з
valueЁBЏ B”
с
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
	regularization_losses

	keras_api
%
#_self_saveable_object_factories
Н

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
Н

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
t
iter
	decay
learning_rateaccumulator1accumulator2accumulator3accumulator4
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
 
≠
	variables
metrics

layers
layer_regularization_losses
trainable_variables
 layer_metrics
!non_trainable_variables
	regularization_losses
 
[Y
VARIABLE_VALUEhidden_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
≠
	variables
"metrics

#layers
$layer_regularization_losses
trainable_variables
%layer_metrics
&non_trainable_variables
regularization_losses
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
≠
	variables
'metrics

(layers
)layer_regularization_losses
trainable_variables
*layer_metrics
+non_trainable_variables
regularization_losses
KI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

,0

0
1
2
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
	-total
	.count
/	variables
0	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

/	variables
ЦУ
VARIABLE_VALUE#Adagrad/hidden_0/kernel/accumulator\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE!Adagrad/hidden_0/bias/accumulatorZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE'Adagrad/output_layer/kernel/accumulator\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE%Adagrad/output_layer/bias/accumulatorZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_layerPlaceholder*'
_output_shapes
:€€€€€€€€€%*
dtype0*
shape:€€€€€€€€€%
Л
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerhidden_0/kernelhidden_0/biasoutput_layer/kerneloutput_layer/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_64668512
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
”
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_0/kernel/Read/ReadVariableOp!hidden_0/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adagrad/hidden_0/kernel/accumulator/Read/ReadVariableOp5Adagrad/hidden_0/bias/accumulator/Read/ReadVariableOp;Adagrad/output_layer/kernel/accumulator/Read/ReadVariableOp9Adagrad/output_layer/bias/accumulator/Read/ReadVariableOpConst*
Tin
2	*
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
!__inference__traced_save_64668676
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_0/kernelhidden_0/biasoutput_layer/kerneloutput_layer/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotalcount#Adagrad/hidden_0/kernel/accumulator!Adagrad/hidden_0/bias/accumulator'Adagrad/output_layer/kernel/accumulator%Adagrad/output_layer/bias/accumulator*
Tin
2*
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
GPU2*0J 8В *-
f(R&
$__inference__traced_restore_64668725≥ґ
Ч
√
#__inference__wrapped_model_64668365
input_layer1
-model_hidden_0_matmul_readvariableop_resource2
.model_hidden_0_biasadd_readvariableop_resource5
1model_output_layer_matmul_readvariableop_resource6
2model_output_layer_biasadd_readvariableop_resource
identityИҐ%model/hidden_0/BiasAdd/ReadVariableOpҐ$model/hidden_0/MatMul/ReadVariableOpҐ)model/output_layer/BiasAdd/ReadVariableOpҐ(model/output_layer/MatMul/ReadVariableOpї
$model/hidden_0/MatMul/ReadVariableOpReadVariableOp-model_hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%»*
dtype02&
$model/hidden_0/MatMul/ReadVariableOp¶
model/hidden_0/MatMulMatMulinput_layer,model/hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
model/hidden_0/MatMulЇ
%model/hidden_0/BiasAdd/ReadVariableOpReadVariableOp.model_hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02'
%model/hidden_0/BiasAdd/ReadVariableOpЊ
model/hidden_0/BiasAddBiasAddmodel/hidden_0/MatMul:product:0-model/hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
model/hidden_0/BiasAddЖ
model/hidden_0/ReluRelumodel/hidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
model/hidden_0/Relu«
(model/output_layer/MatMul/ReadVariableOpReadVariableOp1model_output_layer_matmul_readvariableop_resource*
_output_shapes
:	»$*
dtype02*
(model/output_layer/MatMul/ReadVariableOp«
model/output_layer/MatMulMatMul!model/hidden_0/Relu:activations:00model/output_layer/MatMul/ReadVariableOp:value:0*
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
model/output_layer/SoftmaxЮ
IdentityIdentity$model/output_layer/Softmax:softmax:0&^model/hidden_0/BiasAdd/ReadVariableOp%^model/hidden_0/MatMul/ReadVariableOp*^model/output_layer/BiasAdd/ReadVariableOp)^model/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::2N
%model/hidden_0/BiasAdd/ReadVariableOp%model/hidden_0/BiasAdd/ReadVariableOp2L
$model/hidden_0/MatMul/ReadVariableOp$model/hidden_0/MatMul/ReadVariableOp2V
)model/output_layer/BiasAdd/ReadVariableOp)model/output_layer/BiasAdd/ReadVariableOp2T
(model/output_layer/MatMul/ReadVariableOp(model/output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
≠
†
(__inference_model_layer_call_fn_64668466
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_646684552
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
ц	
я
F__inference_hidden_0_layer_call_and_return_conditional_losses_64668585

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%»*
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
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
Л
Ю
&__inference_signature_wrapper_64668512
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_646683652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
Њ;
Ђ
$__inference__traced_restore_64668725
file_prefix$
 assignvariableop_hidden_0_kernel$
 assignvariableop_1_hidden_0_bias*
&assignvariableop_2_output_layer_kernel(
$assignvariableop_3_output_layer_bias#
assignvariableop_4_adagrad_iter$
 assignvariableop_5_adagrad_decay,
(assignvariableop_6_adagrad_learning_rate
assignvariableop_7_total
assignvariableop_8_count:
6assignvariableop_9_adagrad_hidden_0_kernel_accumulator9
5assignvariableop_10_adagrad_hidden_0_bias_accumulator?
;assignvariableop_11_adagrad_output_layer_kernel_accumulator=
9assignvariableop_12_adagrad_output_layer_bias_accumulator
identity_14ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9о
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ъ
valueрBнB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names™
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesс
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
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

Identity_2Ђ
AssignVariableOp_2AssignVariableOp&assignvariableop_2_output_layer_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_output_layer_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOpassignvariableop_4_adagrad_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5•
AssignVariableOp_5AssignVariableOp assignvariableop_5_adagrad_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6≠
AssignVariableOp_6AssignVariableOp(assignvariableop_6_adagrad_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Э
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Э
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ї
AssignVariableOp_9AssignVariableOp6assignvariableop_9_adagrad_hidden_0_kernel_accumulatorIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10љ
AssignVariableOp_10AssignVariableOp5assignvariableop_10_adagrad_hidden_0_bias_accumulatorIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11√
AssignVariableOp_11AssignVariableOp;assignvariableop_11_adagrad_output_layer_kernel_accumulatorIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ѕ
AssignVariableOp_12AssignVariableOp9assignvariableop_12_adagrad_output_layer_bias_accumulatorIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpь
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13п
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
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
§
Х
C__inference_model_layer_call_and_return_conditional_losses_64668424
input_layer
hidden_0_64668391
hidden_0_64668393
output_layer_64668418
output_layer_64668420
identityИҐ hidden_0/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCall£
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_64668391hidden_0_64668393*
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
GPU2*0J 8В *O
fJRH
F__inference_hidden_0_layer_call_and_return_conditional_losses_646683802"
 hidden_0/StatefulPartitionedCall‘
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_64668418output_layer_64668420*
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
GPU2*0J 8В *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_646684072&
$output_layer/StatefulPartitionedCallЋ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
Х
Р
C__inference_model_layer_call_and_return_conditional_losses_64668455

inputs
hidden_0_64668444
hidden_0_64668446
output_layer_64668449
output_layer_64668451
identityИҐ hidden_0/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallЮ
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_64668444hidden_0_64668446*
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
GPU2*0J 8В *O
fJRH
F__inference_hidden_0_layer_call_and_return_conditional_losses_646683802"
 hidden_0/StatefulPartitionedCall‘
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_64668449output_layer_64668451*
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
GPU2*0J 8В *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_646684072&
$output_layer/StatefulPartitionedCallЋ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
≠
†
(__inference_model_layer_call_fn_64668493
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_646684822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:€€€€€€€€€%
%
_user_specified_nameinput_layer
г
Ѓ
C__inference_model_layer_call_and_return_conditional_losses_64668530

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИҐhidden_0/BiasAdd/ReadVariableOpҐhidden_0/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOp©
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%»*
dtype02 
hidden_0/MatMul/ReadVariableOpП
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_0/MatMul®
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02!
hidden_0/BiasAdd/ReadVariableOp¶
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_0/Reluµ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	»$*
dtype02$
"output_layer/MatMul/ReadVariableOpѓ
output_layer/MatMulMatMulhidden_0/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
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
output_layer/SoftmaxА
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
ж
А
+__inference_hidden_0_layer_call_fn_64668594

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
 *(
_output_shapes
:€€€€€€€€€»*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_hidden_0_layer_call_and_return_conditional_losses_646683802
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€%::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
ц	
я
F__inference_hidden_0_layer_call_and_return_conditional_losses_64668380

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	%»*
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
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€%::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
Ю
Ы
(__inference_model_layer_call_fn_64668561

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_646684552
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
Л(
£
!__inference__traced_save_64668676
file_prefix.
*savev2_hidden_0_kernel_read_readvariableop,
(savev2_hidden_0_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_adagrad_hidden_0_kernel_accumulator_read_readvariableop@
<savev2_adagrad_hidden_0_bias_accumulator_read_readvariableopF
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
ShardedFilenameи
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ъ
valueрBнB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names§
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices»
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_0_kernel_read_readvariableop(savev2_hidden_0_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adagrad_hidden_0_kernel_accumulator_read_readvariableop<savev2_adagrad_hidden_0_bias_accumulator_read_readvariableopBsavev2_adagrad_output_layer_kernel_accumulator_read_readvariableop@savev2_adagrad_output_layer_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
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

identity_1Identity_1:output:0*g
_input_shapesV
T: :	%»:»:	»$:$: : : : : :	%»:»:	»$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	%»:!

_output_shapes	
:»:%!

_output_shapes
:	»$: 
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
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	%»:!

_output_shapes	
:»:%!

_output_shapes
:	»$: 

_output_shapes
:$:

_output_shapes
: 
€	
г
J__inference_output_layer_layer_call_and_return_conditional_losses_64668605

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»$*
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
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
г
Ѓ
C__inference_model_layer_call_and_return_conditional_losses_64668548

inputs+
'hidden_0_matmul_readvariableop_resource,
(hidden_0_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИҐhidden_0/BiasAdd/ReadVariableOpҐhidden_0/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOp©
hidden_0/MatMul/ReadVariableOpReadVariableOp'hidden_0_matmul_readvariableop_resource*
_output_shapes
:	%»*
dtype02 
hidden_0/MatMul/ReadVariableOpП
hidden_0/MatMulMatMulinputs&hidden_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_0/MatMul®
hidden_0/BiasAdd/ReadVariableOpReadVariableOp(hidden_0_biasadd_readvariableop_resource*
_output_shapes	
:»*
dtype02!
hidden_0/BiasAdd/ReadVariableOp¶
hidden_0/BiasAddBiasAddhidden_0/MatMul:product:0'hidden_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_0/BiasAddt
hidden_0/ReluReluhidden_0/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
hidden_0/Reluµ
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes
:	»$*
dtype02$
"output_layer/MatMul/ReadVariableOpѓ
output_layer/MatMulMatMulhidden_0/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
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
output_layer/SoftmaxА
IdentityIdentityoutput_layer/Softmax:softmax:0 ^hidden_0/BiasAdd/ReadVariableOp^hidden_0/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::2B
hidden_0/BiasAdd/ReadVariableOphidden_0/BiasAdd/ReadVariableOp2@
hidden_0/MatMul/ReadVariableOphidden_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
Х
Р
C__inference_model_layer_call_and_return_conditional_losses_64668482

inputs
hidden_0_64668471
hidden_0_64668473
output_layer_64668476
output_layer_64668478
identityИҐ hidden_0/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallЮ
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_0_64668471hidden_0_64668473*
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
GPU2*0J 8В *O
fJRH
F__inference_hidden_0_layer_call_and_return_conditional_losses_646683802"
 hidden_0/StatefulPartitionedCall‘
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_64668476output_layer_64668478*
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
GPU2*0J 8В *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_646684072&
$output_layer/StatefulPartitionedCallЋ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
о
Д
/__inference_output_layer_layer_call_fn_64668614

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallэ
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
GPU2*0J 8В *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_646684072
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
€	
г
J__inference_output_layer_layer_call_and_return_conditional_losses_64668407

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»$*
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
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
Ю
Ы
(__inference_model_layer_call_fn_64668574

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_646684822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€%
 
_user_specified_nameinputs
§
Х
C__inference_model_layer_call_and_return_conditional_losses_64668438
input_layer
hidden_0_64668427
hidden_0_64668429
output_layer_64668432
output_layer_64668434
identityИҐ hidden_0/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCall£
 hidden_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_0_64668427hidden_0_64668429*
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
GPU2*0J 8В *O
fJRH
F__inference_hidden_0_layer_call_and_return_conditional_losses_646683802"
 hidden_0/StatefulPartitionedCall‘
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)hidden_0/StatefulPartitionedCall:output:0output_layer_64668432output_layer_64668434*
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
GPU2*0J 8В *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_646684072&
$output_layer/StatefulPartitionedCallЋ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0!^hidden_0/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€$2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€%::::2D
 hidden_0/StatefulPartitionedCall hidden_0/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
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
StatefulPartitionedCall:0€€€€€€€€€$tensorflow/serving/predict:§l
‘
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
	regularization_losses

	keras_api
5_default_save_signature
*6&call_and_return_all_conditional_losses
7__call__"Й
_tf_keras_networkн{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 37]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_0", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["hidden_0", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "cross_entropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.0005000000237487257, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
Ш
#_self_saveable_object_factories"р
_tf_keras_input_layer–{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 37]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
Ш

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
*8&call_and_return_all_conditional_losses
9__call__"ќ
_tf_keras_layerі{"class_name": "Dense", "name": "hidden_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_0", "trainable": true, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 37}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 37]}}
§

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
*:&call_and_return_all_conditional_losses
;__call__"Џ
_tf_keras_layerј{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 36, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
З
iter
	decay
learning_rateaccumulator1accumulator2accumulator3accumulator4"
	optimizer
,
<serving_default"
signature_map
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 
	variables
metrics

layers
layer_regularization_losses
trainable_variables
 layer_metrics
!non_trainable_variables
	regularization_losses
7__call__
5_default_save_signature
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
": 	%»2hidden_0/kernel
:»2hidden_0/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
	variables
"metrics

#layers
$layer_regularization_losses
trainable_variables
%layer_metrics
&non_trainable_variables
regularization_losses
9__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
&:$	»$2output_layer/kernel
:$2output_layer/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
	variables
'metrics

(layers
)layer_regularization_losses
trainable_variables
*layer_metrics
+non_trainable_variables
regularization_losses
;__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
'
,0"
trackable_list_wrapper
5
0
1
2"
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
	-total
	.count
/	variables
0	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
-0
.1"
trackable_list_wrapper
-
/	variables"
_generic_user_object
4:2	%»2#Adagrad/hidden_0/kernel/accumulator
.:,»2!Adagrad/hidden_0/bias/accumulator
8:6	»$2'Adagrad/output_layer/kernel/accumulator
1:/$2%Adagrad/output_layer/bias/accumulator
е2в
#__inference__wrapped_model_64668365Ї
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
Џ2„
C__inference_model_layer_call_and_return_conditional_losses_64668548
C__inference_model_layer_call_and_return_conditional_losses_64668424
C__inference_model_layer_call_and_return_conditional_losses_64668438
C__inference_model_layer_call_and_return_conditional_losses_64668530ј
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
о2л
(__inference_model_layer_call_fn_64668561
(__inference_model_layer_call_fn_64668493
(__inference_model_layer_call_fn_64668574
(__inference_model_layer_call_fn_64668466ј
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
р2н
F__inference_hidden_0_layer_call_and_return_conditional_losses_64668585Ґ
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
’2“
+__inference_hidden_0_layer_call_fn_64668594Ґ
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
ф2с
J__inference_output_layer_layer_call_and_return_conditional_losses_64668605Ґ
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
ў2÷
/__inference_output_layer_layer_call_fn_64668614Ґ
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
—Bќ
&__inference_signature_wrapper_64668512input_layer"Ф
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
 †
#__inference__wrapped_model_64668365y4Ґ1
*Ґ'
%К"
input_layer€€€€€€€€€%
™ ";™8
6
output_layer&К#
output_layer€€€€€€€€€$І
F__inference_hidden_0_layer_call_and_return_conditional_losses_64668585]/Ґ,
%Ґ"
 К
inputs€€€€€€€€€%
™ "&Ґ#
К
0€€€€€€€€€»
Ъ 
+__inference_hidden_0_layer_call_fn_64668594P/Ґ,
%Ґ"
 К
inputs€€€€€€€€€%
™ "К€€€€€€€€€»≤
C__inference_model_layer_call_and_return_conditional_losses_64668424k<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ ≤
C__inference_model_layer_call_and_return_conditional_losses_64668438k<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ ≠
C__inference_model_layer_call_and_return_conditional_losses_64668530f7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ ≠
C__inference_model_layer_call_and_return_conditional_losses_64668548f7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ К
(__inference_model_layer_call_fn_64668466^<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p

 
™ "К€€€€€€€€€$К
(__inference_model_layer_call_fn_64668493^<Ґ9
2Ґ/
%К"
input_layer€€€€€€€€€%
p 

 
™ "К€€€€€€€€€$Е
(__inference_model_layer_call_fn_64668561Y7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p

 
™ "К€€€€€€€€€$Е
(__inference_model_layer_call_fn_64668574Y7Ґ4
-Ґ*
 К
inputs€€€€€€€€€%
p 

 
™ "К€€€€€€€€€$Ђ
J__inference_output_layer_layer_call_and_return_conditional_losses_64668605]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€»
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Г
/__inference_output_layer_layer_call_fn_64668614P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€»
™ "К€€€€€€€€€$≥
&__inference_signature_wrapper_64668512ИCҐ@
Ґ 
9™6
4
input_layer%К"
input_layer€€€€€€€€€%";™8
6
output_layer&К#
output_layer€€€€€€€€€$