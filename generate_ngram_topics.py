import pandas as pd
import numpy as np
import datetime as dt
from pymongo import MongoClient
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
import pickle
print "Enter ngram_size"
ngram_size = input("ngram_size =" )
print 'Fetching data.....'
def prod_mongo():
# production MongoDB connect
    global db
    client = MongoClient('mongodb://paraprod:paraprod@mmsprod-0.paraprod1.7388.mongodbdns.com:27000/paralign')
    db  = client.paralign

prod_mongo()    
ms = [[x['_id'], x['user'], x['content'], 'message',''] for x in db.message.find()]
th = [[x['_id'], x['user'], x['content'], 'thought', x['mood']] for x in db.thought.find()]
df = pd.DataFrame(th+ms, columns = ['_id', 'user', 'content', 'type', 'mood'])#Combining list of  thoughts and messages

df = df[pd.notnull(df.content)]#eliminating Nan rows    
df.content = df.content.str.strip()
df= df[~df.content.str.contains('[^\x00-\x7F]')]#Remove posts with non-ascii characters
df.drop_duplicates('content', inplace = True)
df.content = df.content.str.replace("\'", "")#Remove apostrophe within words like "don't"
df = df.reset_index()
#Loading MYSQL stopwords
mysql = "a's able about above according accordingly across actually after afterwards again against ain't all allow allows almost alone along already also although always am among amongst an and another any anybody anyhow anyone anything anyway anyways anywhere apart appear appreciate appropriate are aren't around as aside ask asking associated at available away awfully be became because become becomes becoming been before beforehand behind being believe below beside besides best better between beyond both brief but by c'mon c's came can can't cannot cant cause causes certain certainly changes clearly co com come comes concerning consequently consider considering contain containing contains corresponding could couldn't course currently definitely described despite did didn't different do does doesn't doing don't done down downwards during each edu eg eight either else elsewhere enough entirely especially et etc even ever every everybody everyone everything everywhere ex exactly example except far few fifth first five followed following follows for former formerly forth four from further furthermore get gets getting given gives go goes going gone got gotten greetings had hadn't happens hardly has hasn't have haven't having he he's hello help hence her here here's hereafter hereby herein hereupon hers herself hi him himself his hither hopefully how howbeit however i'd i'll i'm i've ie if ignored immediate in inasmuch inc indeed indicate indicated indicates inner insofar instead into inward is isn't it it'd it'll it's its itself just keep keeps kept know known knows last lately later latter latterly least less lest let let's like liked likely little look looking looks ltd mainly many may maybe me mean meanwhile merely might more moreover most mostly much must my myself name namely nd near nearly necessary need needs neither never nevertheless new next nine no nobody non none noone nor normally not nothing novel now nowhere obviously of off often oh ok okay old on once one ones only onto or other others otherwise ought our ours ourselves out outside over overall own particular particularly per perhaps placed please plus possible presumably probably provides que quite qv rather rd re really reasonably regarding regardless regards relatively respectively right said same saw say saying says second secondly see seeing seem seemed seeming seems seen self selves sensible sent serious seriously seven several shall she should shouldn't since six so some somebody somehow someone something sometime sometimes somewhat somewhere soon sorry specified specify specifying still sub such sup sure t's take taken tell tends th than thank thanks thanx that that's thats the their theirs them themselves then thence there there's thereafter thereby therefore therein theres thereupon these they they'd they'll they're they've think third this thorough thoroughly those though three through throughout thru thus to together too took toward towards tried tries truly try trying twice two un under unfortunately unless unlikely until unto up upon us use used useful uses using usually value various very via viz vs want wants was wasn't way we we'd we'll we're we've welcome well went were weren't what what's whatever when whence whenever where where's whereafter whereas whereby wherein whereupon wherever whether which while whither who who's whoever whole whom whose why will willing wish with within without won't wonder would wouldn't yes yet you you'd you'll you're you've your yours yourself yourselves zero"
mysql_sw = mysql.split(" ")
mysql_sw = [word.replace("\'", "") for word in mysql_sw]
#Define the models


def run_NMF(docs,vect,model):#,**kwargs):
    
    """Create Prepared Data from sklearn's vectorizer and Latent Dirichlet
    Application
    
    Parameters
    ----------
    docs : Pandas Series.
        Documents to be passed as an input.
    vect : Scikit-Learn Vectorizer (CountVectorizer,TfIdfVectorizer).
        vectorizer to convert documents into matrix sparser
   model : sklearn.decomposition.NMF
    
   
    
    
    Returns
    -------
    vect : sklearn's Vectorizer.
    model :sklearn's Topic Modeler: LDA or NMF.
    doc_topic_dists: Topic distribution for each document
    good_rows: Indices of non-zero rows in doc_topic_dists
    """
    
    
    vected = vect.fit_transform(docs)
    dtd  = normalize(model.fit_transform(vected), norm = 'l1')
    good_rows = ~(dtd==0).all(1)
    doc_topic_dists = dtd[good_rows]#Eliminating zero rows (docs that cannot be asscoiated with any of the topics)
    return (model,vect,doc_topic_dists, good_rows)



def generate_ngram_collection(dframe, ngram_size, min_df_, n_components, alpha, top_n_topics):	
	corpus = dframe.content
	print "Running NMF to generate topics....."
	tfidf_vectorizer = TfidfVectorizer(analyzer='word', max_df=0.95, min_df= min_df_, ngram_range=(ngram_size,ngram_size), stop_words = 		mysql_sw)
	tf_vectorizer = CountVectorizer(analyzer='word', max_df=0.95, min_df= min_df_, ngram_range=(ngram_size,ngram_size), stop_words = 		mysql_sw)
	nmf_model = NMF(n_components= n_components, random_state=42, alpha=alpha, l1_ratio= 0.5)
	model, vect, doc_topic_dists, good_rows = run_NMF(corpus, tfidf_vectorizer, nmf_model)
	docs = dframe[good_rows]
	dtd = doc_topic_dists
	topic_weights = dtd.sum(axis = 0)
	top_idx = topic_weights.argsort()[::-1]# Sort the topic indices in decreasing order of their relative corpus-wide weight
	topic_term_dists = normalize(model.components_, norm = 'l1')
	topics = [(vect.get_feature_names()[topic.argmax()],topic.argmax()) for topic_ix, topic in enumerate(topic_term_dists)]# list of tuples of (topic_name, id)
	topics = [topics[i] for i in top_idx]
	topics = topics[:top_n_topics]
	topics = [i for ix,i in enumerate(topics) if topics[ix-1]!= i]
	top_n_topics = len(topics)
	#def is_ascii(s):
	    #return all(ord(c) < 128 for c in s)
	#topics = [topic for topic in topics if is_ascii(topic[0])] #eliminate any non-ascii topic names
	topic_names = [t[0] for t in topics]
	topic_length = top_n_topics
	print "The top",top_n_topics,"topic_names are:"
	print topic_names
        return topics, topic_length, topic_names, top_idx, docs, dtd

def write_to_dict(docs,mood, dtd, topic_length, topics, tix, t_names):#Appends to dict:-{mood:{topic:{thoughts:[], msgs]}}}	
	moods_dict ={}	
	moods_dict["mood"] = mood
	moods_dict["topics"] = {str(t):{'thoughts':[],'msgs':[]} for t in t_names}
	doc_topics = np.array(dtd[:,[i for i in tix[:topic_length]]])#Keeping only the top_n topics
	topic_docs = doc_topics.T
	for tix, topic_dist in enumerate(topic_docs):
	    sorted_indices = topic_dist.argsort()[::-1]
	    for i,j in enumerate(sorted(topic_dist, reverse = True)):
		if j >0.3:
		    index = sorted_indices[i]		    
		    if (docs.iloc[index]['type'] == 'thought'):
	            	moods_dict["topics"][topics[tix][0]]['thoughts'].append(docs.iloc[index]._id)
		    elif (docs.iloc[index]['type'] == 'message'):
	            	moods_dict["topics"][topics[tix][0]]['msgs'].append(docs.iloc[index]._id)
	return moods_dict

    
def make_collection(dict1):#Creates a master dict for a mongodb collection
    out_dict= dict1
    out_dict['date'] = str(dt.datetime.now())
    out_dict['algorithm'] = 'NMF'
    return out_dict

def push_to_db(coll):
#collection = make_collection(dict_)
	print "Pushing collection to the db......"
	if ngram_size ==1: 
		result = db.topics_uni.insert_one(coll)
		#pickle.dump(collection, open( "topics_uni.p", "wb" ))
	if ngram_size ==2: 
		result = db.topics_bi.insert_one(coll)
		#pickle.dump(collection, open( "topics_bi.p", "wb" ))
	if ngram_size ==3: 	
		result = db.topics_tri.insert_one(coll)
		#pickle.dump(collection, open( "topics_tri.p", "wb" ))

#print "Select parameters: ngram size, n_components, alpha, top_n_topics:"
params_uni = {'all': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7},
 'good': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7},
 'happy': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7},
 'mad': {'alpha': 0.2, 'min_df': 5, 'n_components': 7, 'top_n_topics': 7},
 'nervous': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7},
 'neutral': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7},
 'peaceful': {'alpha': 0.2,
  'min_df': 5,
  'n_components': 10,
  'top_n_topics': 7},
 'sad': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7}}

params_bi = {'all': {'alpha': 0.2, 'min_df': 5, 'n_components': 7, 'top_n_topics': 7},
 'good': {'alpha': 0.1, 'min_df': 4, 'n_components': 7, 'top_n_topics': 7},
 'happy': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7},
 'mad': {'alpha': 0.2, 'min_df': 3, 'n_components': 7, 'top_n_topics': 6},
 'nervous': {'alpha': 0.2, 'min_df': 3, 'n_components': 7, 'top_n_topics': 6},
 'neutral': {'alpha': 0.2, 'min_df': 5, 'n_components': 10, 'top_n_topics': 7},
 'peaceful': {'alpha': 0.2,
  'min_df': 4,
  'n_components': 10,
  'top_n_topics': 7},
 'sad': {'alpha': 0.2, 'min_df': 5, 'n_components': 8, 'top_n_topics': 7}}

params_tri = {'all': {'alpha': 0.3, 'min_df': 5, 'n_components': 7, 'top_n_topics': 5}, 'good': {'alpha': 0.2, 'min_df': 2, 'n_components':5 , 'top_n_topics': 4}, 'happy': {'alpha': 0.2, 'min_df': 2, 'n_components': 6, 'top_n_topics': 5}, 'mad': {'alpha': 0.2, 'min_df': 2, 'n_components': 4, 'top_n_topics': 4}, 'nervous': {'alpha': 0.2, 'min_df': 2, 'n_components': 6, 'top_n_topics': 5}, 'neutral': {'alpha': 0.2, 'min_df': 3, 'n_components': 7, 'top_n_topics': 5}, 'peaceful': {'alpha': 0.2, 'min_df': 2, 'n_components': 7, 'top_n_topics': 6}, 'sad': {'alpha': 0.2, 'min_df': 2, 'n_components': 7, 'top_n_topics': 6}}
params = {1: params_uni, 2: params_bi, 3: params_tri}

min_df = params[ngram_size]['all']['min_df']
n_components = params[ngram_size]['all']['n_components']
alpha = params[ngram_size]['all']['alpha']
top_n_topics = params[ngram_size]['all']['top_n_topics']


print "Generating topics for all thoughts + messages...."
topics_, topic_length_, topic_names_, topic_idx_, docs_, dtd_ = generate_ngram_collection(df, ngram_size, min_df, n_components, alpha, top_n_topics)
dict_ = write_to_dict(docs_, "all", dtd_, topic_length_, topics_, topic_idx_, topic_names_)
collection = make_collection(dict_)
push_to_db(collection)

moods = ['sad', 'neutral', 'good', 'peaceful', 'happy', 'mad', 'nervous']
print "Generating topics for thoughts grouped by mood"
for mood in moods:
	dfm = df[df.mood == mood].reset_index()
	print "mood =",mood
	min_dfm = params[ngram_size][mood]['min_df']
	n_components = params[ngram_size][mood]['n_components']
	alpha = params[ngram_size][mood]['alpha']
	top_n_topics = params[ngram_size][mood]['top_n_topics']
	topics, topic_length, topic_names, topic_idx, docs, dtd = generate_ngram_collection(dfm, ngram_size, min_dfm, n_components, 		alpha, top_n_topics)
	dict_ = write_to_dict(docs, mood, dtd,topic_length, topics, topic_idx, topic_names)
	collection = make_collection(dict_)
	push_to_db(collection)
