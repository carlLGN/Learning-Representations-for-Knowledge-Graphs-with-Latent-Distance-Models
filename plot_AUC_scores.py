import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# chose number of epochs
epoch_numbers=np.arange(1000,16000,1000)
roc_auc_test=np.array([0.29034500799077234, 0.3847267268160957, 0.45123621825146243, 0.4957287905883826, 0.5291590522696532, 0.5522746610252844, 0.5621663525959977, 0.5638053308053357, 0.5571976716959265, 0.5440252220569973, 0.5266243187962738, 0.5107797364965616, 0.49861772617885686, 0.49533809253012584, 0.49403783286827063])
roc_auc_train=np.array([0.185217220463138, 0.34861175992438564, 0.45930773842155015, 0.5329761637523629, 0.5928756767485823, 0.6326168175803403, 0.6530295368620037, 0.6638204529773156, 0.6655197334593572, 0.656055768194707, 0.6447879853497165, 0.6304135590737241, 0.6150738060018903, 0.6143516935255199, 0.6147049068998109])
pr_auc_test=np.array([0.380564518725965, 0.413444990685923, 0.4486045565876781, 0.48000615450154455, 0.5072291675827374, 0.5294440479184629, 0.5420511387991905, 0.548363020623874, 0.5444162306694073, 0.5302148128162326, 0.5134105600939998, 0.49892380163745925, 0.48542226897777135, 0.4787742668396906, 0.47640511439324096])
pr_auc_train=np.array([0.33960763018487056, 0.3910294806118196, 0.4462588034786169, 0.49750917260817135, 0.5505871291947154, 0.5954012265186666, 0.624373248576923, 0.6414562582706796, 0.6405204680610089, 0.6206449582949245, 0.6008215843114599, 0.5781007533400856, 0.5575559087998063, 0.5499739093014275, 0.5482431485861213])

fig, ax = plt.subplots()
ax.plot(epoch_numbers,pr_auc_test, color='blue',label='Test set')
ax.plot(epoch_numbers,pr_auc_train, color='red',label='Training set')
plt.legend(loc='lower right')
plt.title('PR-AUC')
plt.show()


# chose value for alpha a2p
#alpha0_ROC=np.array([0.5670462454346294, 0.5649060622876241, 0.5671330877513916, 0.5672878828338057])
#alpha0_PR=np.array([0.5501755035283742, 0.5483292232948821, 0.5484239379485808, 0.5505432571815033])

#alpha01_ROC=np.array([0.5685943782081765, 0.5667427907244806, 0.5683426586303253, 0.5690988394368225])
#alpha01_PR=np.array([0.5505232622442622, 0.5490225899958393, 0.5487521087507, 0.5516080108998331])

#alpha02_ROC=np.array([0.5691440798571936, 0.5684599391798089, 0.5688986804683683, 0.5704556790980888])
#alpha02_PR=np.array([0.550995781097229, 0.549581353075824, 0.5483593328778974, 0.5519991645783394])

#alpha03_ROC=np.array([0.570085091787089, 0.5698062891328615, 0.5699463370301888, 0.5707147994216475])
#alpha03_PR=np.array([0.5512750926436042, 0.550041680301809, 0.5492248271810499, 0.552145072820567])

#alpha04_ROC = np.array([0.570762297035808, 0.5718762685865403, 0.5709758413537375, 0.5714791052903045])
#alpha04_PR = np.array([0.5518127377578016, 0.5512742389805283, 0.5496527242833567, 0.5522413276321434])

#alpha05_ROC = np.array([0.5717424847152843, 0.5732214460798948, 0.5718284107868193, 0.5728128587281085])
#alpha05_PR = np.array([0.5522470726573213, 0.5517034129548347, 0.5498928876955466, 0.5527972416052286])

#alpha06_ROC = np.array([0.572929548638255, 0.5745823496647111, 0.5720053294318034, 0.5736511291193493])
#alpha06_PR = np.array([0.5526340511259765, 0.553221543178775, 0.5502871737451819, 0.5534811365022513])

#alpha07_ROC = np.array([0.5727523139025923, 0.5753002913897471, 0.5722440700293263, 0.5738596744701574])
#alpha07_PR = np.array([0.551545285924536, 0.5536010244552307, 0.5509113880915026, 0.5537447386747278])

#alpha08_ROC = np.array([0.573523638051877, 0.5740439924683788, 0.571515223407953, 0.573731046683049])
#alpha08_PR = np.array([0.5515837567282802, 0.5521515142807484, 0.5495274976953857, 0.5523485774675354])

#alpha09_ROC = np.array([0.5723100728728912, 0.5735416092690553, 0.572810567882962, 0.5748470964951821])
#alpha09_PR = np.array([0.5509815616496755, 0.5514608932755012, 0.5509273995363281, 0.5518450165137865])

#alpha1_ROC = np.array([0.5137369602058675, 0.5140104603345244, 0.5160076057191406, 0.5176769003200795])
#alpha1_PR = np.array([0.5095009746671452, 0.5093249637340331, 0.5113718973823227, 0.5132976144028323])

#ROC = np.array([alpha0_ROC,alpha01_ROC,alpha02_ROC,alpha03_ROC,alpha04_ROC,alpha05_ROC,alpha06_ROC,alpha07_ROC,alpha08_ROC,alpha09_ROC,alpha1_ROC])
#PR = np.array([alpha0_PR,alpha01_PR,alpha02_PR,alpha03_PR,alpha04_PR,alpha05_PR,alpha06_PR,alpha07_PR,alpha08_PR,alpha09_PR,alpha1_PR])

# chose value for alpha a2p
alpha0_ROC=np.array([0.5139942504180098, 0.5129974611502941, 0.5126721524022754, 0.5104210402738061])
alpha0_PR=np.array([0.5100390276908593, 0.5099537342252727, 0.5095157928941763, 0.5075276938967743])

alpha01_ROC=np.array([0.5392807594222249, 0.5366336654243505, 0.5392426651938667, 0.5380915588683567])
alpha01_PR=np.array([0.5021775478557212, 0.4993998856635743, 0.5030544973621494, 0.5010069555408898])

alpha02_ROC=np.array([0.5388720377568939, 0.5372465995338935, 0.5393936636201688, 0.5379187939129974])
alpha02_PR=np.array([0.5014847441212367, 0.499359731352131, 0.5022005886789667, 0.5003596766858132])

alpha03_ROC=np.array([0.5396144682973034, 0.5385998690636926, 0.5401718211077174, 0.5389202644666113])
alpha03_PR=np.array([0.5012263915962345, 0.4997036198790358, 0.5019039105028505, 0.5003858536680583])

alpha04_ROC = np.array([0.5415652181575057, 0.5398345358859973, 0.5407575607217756, 0.5400115711698242])
alpha04_PR = np.array([0.5015050614121187, 0.499742667215619, 0.5010616397389794, 0.4998290288865371])

alpha05_ROC = np.array([0.5433431701256592, 0.5407354909440404, 0.5423915148703019, 0.542072439477444])
alpha05_PR = np.array([0.5021168765387254, 0.4995653981665293, 0.5013374478071241, 0.5003995136850902])

alpha06_ROC = np.array([0.5452066488531859, 0.544226874332557, 0.5413606012276335, 0.5437195695730055])
alpha06_PR = np.array([0.5019894687983361, 0.502769272545439, 0.4991299267581509, 0.5017529503341484])

alpha07_ROC = np.array([0.5466371720237961, 0.5463159588664396, 0.543913899507996, 0.5448708821862023])
alpha07_PR = np.array([0.5022988440091156, 0.5035365915191942, 0.5001767035260741, 0.5016427786403401])

alpha08_ROC = np.array([0.5500307229279107, 0.5495923874805766, 0.5473200790322637, 0.5475481430713284])
alpha08_PR = np.array([0.5041376452392207, 0.5051129974149426, 0.5023002794223805, 0.5030205661922255])

alpha09_ROC = np.array([0.5503401700596766, 0.5549406758956679, 0.5566398698830557, 0.5530312391607088])
alpha09_PR = np.array([0.503728154626985, 0.5068414989620861, 0.508509936956099, 0.5068895232028332])

alpha1_ROC = np.array([0.5593023754779938, 0.5626639631162553, 0.5630675821371972, 0.5622581534009768])
alpha1_PR = np.array([0.508916482762843, 0.5115968515338944, 0.512051142903845, 0.512020086549525])
ROC = np.array([alpha0_ROC,alpha01_ROC,alpha02_ROC,alpha03_ROC,alpha04_ROC,alpha05_ROC,alpha06_ROC,alpha07_ROC,alpha08_ROC,alpha09_ROC,alpha1_ROC])
PR = np.array([alpha0_PR,alpha01_PR,alpha02_PR,alpha03_PR,alpha04_PR,alpha05_PR,alpha06_PR,alpha07_PR,alpha08_PR,alpha09_PR,alpha1_PR])


def plot_AUC_combined(ROC,PR):
    mean_ROC= [np.mean(ROC[i,:]) for i in range(len(ROC))]
    std_ROC = [ROC[i,:].std() for i in range(len(ROC))]
    CI_ROC = st.t.interval(confidence=0.95, df=len(ROC[0,:])-1, loc=mean_ROC, scale=std_ROC)

    mean_PR = [np.mean(PR[i, :]) for i in range(len(PR))]
    std_PR = [PR[i, :].std() for i in range(len(PR))]
    CI_PR = st.t.interval(confidence=0.95, df=len(PR[0, :]) - 1, loc=mean_PR, scale=std_PR)

    fig, ax = plt.subplots()

    ax.plot(np.arange(0,1.1,0.1),mean_ROC, color='blue',label='ROC')
    ax.fill_between(np.arange(0,1.1,0.1), CI_ROC[0], CI_ROC[1], alpha=.4, color='purple',label='CI ROC')

    ax.plot(np.arange(0, 1.1, 0.1), mean_PR, color='red', label='PR')
    ax.fill_between(np.arange(0, 1.1, 0.1), CI_PR[0], CI_PR[1], alpha=.4, color='orange', label='CI PR')
    plt.legend(loc='upper left')
    plt.show()

print(plot_AUC_combined(ROC,PR))
