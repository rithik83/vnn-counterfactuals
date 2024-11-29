from maraboupy import Marabou
import numpy as np
from torchvision import datasets, transforms

train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

train_set_array = train_set.data.numpy()
test_set_array = test_set.data.numpy()

print(train_set_array[0:1].flatten().shape)

options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=10)

filename = 'models/classically_trained.onnx'
# filename = 'models/adv_pgd_strong.onnx'
network = Marabou.read_onnx(filename)

inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]

# print("inputVars shape: ", inputVars.shape)
# print("outputVars shape: ", outputVars.shape)
# print("outputVars: ", outputVars)

epsilon = 0.1
index = 4521
# image = train_set_array[index: index + 1].flatten() / 255
# correct_class = train_set.targets.numpy()[index]

image = [0.00013171235787012847, 0.0, 0.0, 0.0, 3.1564985579279896e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.370944796188269e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7386946274200456e-5, 0.0, 0.0, 1.782004073902499e-6, 2.6061564267365614e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.602945644663123e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00022927632235223428, 0.0, 0.0, 0.01223655502855458, 0.32006712893766337, 0.9922414579272473, 0.2854008451364638, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0797663450575784e-5, 0.0, 0.0, 8.128793429023063e-5, 0.00011356492877894198, 0.00012293086474528538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001311419462581398, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24597450612119226, 0.9873520979030249, 0.9875034435918398, 0.17161162317215553, 0.0, 0.0, 0.0, 0.0, 0.24783322162095042, 0.00016471472517878285, 0.0, 0.00010950097675959114, 3.33798423753251e-5, 0.0, 8.056315596149943e-5, 0.0005104940326418728, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.728307099836877, 0.9888501721390277, 0.9876170190042041, 0.03982592516530541, 0.0, 0.0, 0.0, 0.00013900899084546837, 0.0, 0.2092473912477519, 0.00027280041340418395, 0.0004594187995280663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.775343237881316e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.022318565083028388, 0.02768039122770018, 0.8402240364096819, 0.9907203779633617, 0.9882334534856745, 0.748446089765391, 0.019989300003028156, 2.7541157101040882e-5, 0.000501365159470879, 0.0, 0.0002131800588358601, 0.0, 4.551543324851082e-5, 0.25096245132420364, 0.0, 0.0, 0.0, 0.00010923304605512386, 0.00017111952688537714, 0.0, 0.0, 0.0, 5.3868529357714584e-5, 0.00047921055120241367, 0.0007452278030541493, 0.00017425827845363523, 0.0003992056659335502, 0.0, 0.03967206835636318, 0.6395470701370385, 0.9880013581784557, 0.9931285619302483, 0.8192938147151534, 0.0586653421907423, 0.00023364485905403854, 0.0, 0.0005529121208383003, 0.0005740965159930057, 0.00028489153794362205, 0.07532541434449948, 0.0, 0.0003288517915279954, 0.0005225435023021419, 0.0, 1.652305468269333e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00023592466932313984, 0.08763014619920825, 0.0, 0.0, 0.0, 0.32675289951603653, 0.9895781927655817, 0.9873947375092004, 0.9487627950732754, 0.29071097817545727, 0.0, 0.17273800385012977, 0.023652592347100446, 0.25139684202582385, 0.004979255216730008, 0.0007201773080378189, 0.0, 0.0, 0.00020943097297276836, 0.0006498173199361191, 0.0, 2.6094617714989e-5, 0.0, 0.00033869000399135987, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0006578465830898495, 0.0, 0.016367274421004708, 0.46249535234708006, 0.9567383864379673, 0.9894812681804378, 0.9682389854531738, 0.26309569313587944, 0.3902097447807208, 0.06771792277365771, 0.19992509688854557, 0.00011625251924851911, 0.0, 0.0, 0.0, 0.00021924232937453782, 0.0, 0.08270159619794411, 1.0, 0.872023481723341, 0.0, 0.0, 0.00018045173592327047, 0.0001920944512676215, 0.0, 0.0, 0.0, 0.0, 0.0004418737850064644, 0.0, 0.42365475892860743, 0.9874931921704391, 0.9876826994509025, 0.937746540443292, 0.2747154308285771, 0.000495936883817194, 0.0002528953233413631, 0.0006953987683118612, 0.000402822342584841, 0.00023867263244028438, 0.0007476616211533838, 0.020141226628248937, 0.00033980644884286447, 0.266731947040295, 0.0, 0.9444846691914215, 0.00039381765902362514, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00047514990392301116, 0.0006253491241295706, 0.06622353923720473, 0.8422162447279988, 0.9868358703535022, 0.9360849657146995, 0.13261156995195797, 0.00023328217339440018, 0.0, 0.0, 2.4226680397987366e-5, 0.00044412828610802536, 0.0, 0.0, 0.0007350664836849319, 0.00046346900073513097, 0.15579646911442088, 0.0, 0.0006969175916310633, 0.00021889708700473422, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00018489417502132692, 0.038928692390445004, 0.73995597773061, 0.9869149395550896, 0.9865153890145831, 0.6278268363898003, 0.0, 0.0, 0.0, 0.11400346843658427, 0.4353928473565676, 0.8476080819400699, 0.8635843759132292, 0.7149429310765574, 0.22704564333851912, 0.1563203621559127, 0.00020144011998490894, 0.49838074226740875, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0004980682718723983, 0.0, 0.33236818762132186, 0.9906356019142876, 0.9921278389961448, 0.7043867330873657, 0.0, 0.0, 0.49258675254845813, 1.0, 0.9919923170068315, 0.9923825824415103, 0.9933592610631286, 0.9934825903204193, 0.9930238618946593, 0.9582952129742776, 0.052013353625146995, 0.0, 0.0, 0.00014021633510310493, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00042136581414524703, 0.0, 0.10575748750296808, 0.8314950583440185, 0.9885326683186082, 0.8976798520597028, 0.19229360971784218, 0.16075342696791842, 0.39180773830234455, 0.953007367271157, 0.9913274402330607, 0.987777460688372, 0.8859321973969406, 0.6915365388606977, 0.5802087685168478, 0.9334176118493931, 0.9890167743971223, 0.3894277568921231, 0.0, 0.0, 0.0, 0.000570552256249357, 0.3103853815975898, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.8338451233576055e-5, 0.3920864802168973, 0.9882292340343984, 0.9880494807143406, 0.4793297758183626, 0.20896459183576108, 0.9259012044291384, 0.9889615414206242, 0.8206481032687593, 0.3618287230385537, 0.17221384682988133, 0.10632130826625497, 0.0, 0.0, 0.7580613176654654, 0.988761852818275, 0.31146574463955773, 0.0, 0.0, 0.0, 0.0004656957746628904, 0.0, 0.0, 4.788903097505682e-5, 0.0, 0.0, 0.00032891573760025495, 0.0, 0.0, 0.39222377571838224, 0.9886311416752189, 0.9882151257006235, 0.6198547426132858, 0.9334065429749376, 0.9728098365993699, 0.769240441841185, 0.07887729098672581, 0.00037705426302636624, 0.00033248613381147156, 0.0, 0.0, 0.0, 0.7753472820644354, 0.9885606795656172, 0.38808666062975933, 0.0, 0.0, 0.0, 0.0006006463012454334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003001956105435966, 0.39169345637360664, 0.9883972070373541, 0.9890929524878563, 0.989239779467347, 0.9881678053078752, 0.7973417000903368, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04938553651494359, 0.822635333986044, 0.9799092604691813, 0.3025980188403474, 0.0, 0.0, 0.0, 0.0, 0.0006757576165909995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3913061245871133, 0.9881230469331053, 0.9886747976193965, 0.9881213579773664, 0.8033805691850859, 0.08639484374432584, 0.0003540066875757475, 0.00019250508412369527, 0.0, 0.0, 0.0, 0.18455251853262658, 0.6739176374398806, 0.987608499436939, 0.796615941793488, 0.0, 0.0, 0.0, 0.0, 0.0005905416479872656, 0.0004511629791741072, 0.000168109152764373, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00023363718555629022, 0.37569552020141556, 0.9870995733475212, 0.987489755329078, 0.838242341924325, 0.10265673231994768, 0.0, 0.0, 0.019348398934051988, 0.019898185294163966, 0.011530287760020956, 0.404415035795863, 0.8986602763658894, 0.9884062020816635, 0.8039549440608709, 0.08186022005023562, 0.0, 0.0, 0.0006831122002040502, 0.0, 0.0005167950680042849, 0.00029706908662774367, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.686739965588818, 0.9888587778200842, 0.987333224876757, 0.4352406125716363, 0.6026077841047157, 0.6044839885130118, 0.771680039782276, 0.7762894921405693, 0.706193078169036, 0.9877459065961267, 0.9878831957370534, 0.8747719842281262, 0.1768597702997964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027599097266753397, 0.00019164679724781308, 3.377113080205163e-5, 0.0, 0.00025369987979502185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17269601913486643, 0.8865988229815793, 0.9889503976970274, 0.9871496285846839, 0.9881614311697674, 0.9877836148446734, 0.9881561434138635, 0.9924767503508358, 0.9888118917961992, 0.9884348103498, 0.4751008840372438, 0.09004826566167881, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00036932453758709016, 6.375780496819063e-5, 0.0005725547445763369, 0.0002772313397144899, 0.0, 0.01412997451126177, 0.0004597204737365246, 0.0002943713450804353, 0.08192630985889306, 0.4671096612779783, 0.7648099601296615, 0.988479263840126, 0.9886207741624831, 0.798847138627108, 0.5604920514339279, 0.5558471950142492, 0.20056990583726625, 0.006995800347317444, 0.0002219667191639019, 0.0, 0.0, 0.0003654235498288472, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0004392472732433817, 0.31001944676336946, 0.000490762136381818, 4.402865226802534e-5, 0.0008601896502332237, 0.0005226260684139561, 0.0002765445393379196, 0.0, 0.00021838999855390285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00010620244383972022, 0.0, 0.0, 0.00037354421583586374, 0.0, 0.0002475887184555177, 2.4384571588598193e-7, 0.0, 2.391096813880722e-6, 0.0, 2.9311408798093908e-5, 0.0005063976388555603, 0.0, 0.0, 0.0, 0.0, 0.0006540778980706819, 0.0003239038072933909, 0.0004108275523321936, 0.0, 2.9001837855503255e-5, 0.00028867851635732225, 0.0, 0.0, 0.0, 0.00021541372952924576, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003002818686290991, 0.12430537416005928, 0.00039428862946806477, 0.0001544761795230443, 0.0, 0.0, 0.0, 6.461764005507576e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.014185754000203e-5, 0.3749693561978487, 0.3633652785391708, 0.23519534431109093, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00027716592649085214, 0.00045978288308106134, 0.00021491935222002213, 0.0004952412073180312, 0.0003000232314661844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00013711314354623028, 0.0, 0.0, 6.848382217867765e-5, 0.00018417650230549037, 0.0005852128513652133, 0.0, 0.00013796374587400352, 0.00023130189401854295, 0.26568286012034, 0.20148487195751438, 0.0002935803839354776, 0.0, 0.00037745093925332186, 0.0006310673927146127, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.936980795562704e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00018436075806675946, 0.003182835357178467, 4.187943838769565e-5, 0.0, 0.0, 0.00037604691569868013, 0.0, 0.0004490629130486923, 0.0, 0.0, 0.42679899836153, 0.0005561253135492733, 0.0, 0.0, 0.0002462058379023802, 0.0, 0.00012799740834452679, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0766078089072836e-5, 0.0, 0.0, 0.0, 0.00031765849535076996, 0.0004307675617383211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6417624487075945e-5, 0.07636605297823204, 0.0, 0.0, 0.0, 0.00020775644027162343, 0.000210027707998961, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
correct_class = 8
print("correct class: ", correct_class)


for i in range(len(inputVars)):
    network.setLowerBound(inputVars[i], max(image[i] - epsilon, 0))
    network.setUpperBound(inputVars[i], min(image[i] + epsilon, 1))

margin = -0.00001

for i in range(len(outputVars)):
  print("i: ", i)
  if i != correct_class:
    network.addMaxConstraint(set(outputVars), outputVars[i])
    network.addInequality([outputVars[correct_class], outputVars[i]], [1, -1], margin)
    exit_code, vals, stats = network.solve(verbose = False, options = options)

    print("satisfiability? ", exit_code)

    # if solution found, break
    if len(vals) > 0:
      for j, var in enumerate(outputVars):
        print(f"output {j}: {vals[var]}")
      print(f"maxclass: {i}")
      inputPoint = np.zeros((1, 784))
      for j, var in enumerate(inputVars):
        inputPoint[0][j] = vals[j]
      
      meval = network.evaluateWithMarabou(inputPoint, options = options)[0]
      onnxeval = network.evaluateWithoutMarabou(inputPoint)[0]

      print("marabou eval: ", meval)
      print("onnx eval: ", onnxeval)
      
      break










# print("\nConvolutional Network with Max Pool Example")
# filename = 'resources/classic_cnn.onnx'
# network = Marabou.read_onnx(filename)

# # # %%
# # Get the input and output variable numbers; [0] since first dimension is batch size
# inputVars = network.inputVars[0]
# outputVars = network.outputVars[0]

# # %% 
# # Test Marabou equations against onnxruntime at an example input point
# inputPoint = np.zeros(inputVars.shape)
# print("inputvars shape: ", inputVars.shape)
# marabouEval = network.evaluateWithMarabou([inputPoint], options = options)[0]
# onnxEval = network.evaluateWithoutMarabou([inputPoint])[0]

# # # %%
# # The two evaluations should produce the same result
# print("Marabou Evaluation:")
# print(marabouEval)
# print("\nONNX Evaluation:")
# print(onnxEval)
# print("\nDifference:")
# print(onnxEval - marabouEval)
# assert max(abs(onnxEval - marabouEval).flatten()) < 1e-3


# print("\nConvolutional Network Example")
# filename = 'resources/KJ_TinyTaxiNet.onnx'
# network = Marabou.read_onnx(filename)

# # %%
# # Get the input and output variable numbers; [0] since first dimension is batch size
# inputVars = network.inputVars[0][0]
# outputVars = network.outputVars[0][0]

# # %%
# # Setup a local robustness query
# delta = 0.03
# for h in range(inputVars.shape[0]):
#     for w in range(inputVars.shape[1]):
#         network.setLowerBound(inputVars[h][w][0], 0.5-delta)
#         network.setUpperBound(inputVars[h][w][0], 0.5+delta)

# # %%
# # Set output bounds
# network.setLowerBound(outputVars[0], 6.0)

# # %%
# # Call to Marabou solver (should be SAT)
# print("Check query with less restrictive output constraint (Should be SAT)")
# exitCode, vals, stats = network.solve(options = options)
# assert( exitCode == "sat")
# assert len(vals) > 0

# print("vals: ", vals)

# # %%
# # Set more restrictive output bounds
# network.setLowerBound(outputVars[0], 6.55)

# # %%
# # Call to Marabou solver (should be UNSAT)
# print("Check query with more restrictive output constraint (Should be UNSAT)")
# exitCode, vals, stats = network.solve(options = options)
# assert( exitCode == "unsat")
# assert len(vals) == 0