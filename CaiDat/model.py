import numpy as np

class HMM:
    def __init__(self, states, observations, start_prob , trans_prob,  em_prob):
        # Initialize start, em and trans_prob 
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.em_prob = em_prob

        self.generate_obs_map()
        self.generate_state_map()

        # Convert everything to lists
        self.states=list(self.states)
        self.observations=list(self.observations)

        # Dimension Check
        s_len = len(states)
        o_len = len(observations)

        if( (s_len,o_len)!= self.em_prob.shape ):
            print("Input 1 has incorrect dimensions, Correct dimensions is ({},{})".format(s_len,o_len))
            return None

        if( (s_len,s_len)!= self.trans_prob.shape ):
            print("Input 2 has incorrect dimensions, Correct dimensions is ({},{})".format(s_len,s_len))
            return None
        if( s_len!= (self.start_prob).shape[1]):
            print("Input 3 has incorrect dimensions, Correct dimensions is,",s_len)

        # No negative numbers
        if(not( (self.start_prob>=0).all() )):
            print("Negative probabilities are not allowed")

        if(not( (self.em_prob>=0).all() )):
            print("Negative probabilities are not allowed")

        if(not( (self.trans_prob>=0).all() )):
            print("Negative probabilities are not allowed")
        tmp2 = [ 1 for i in range(s_len) ]
        summation = np.sum(em_prob,axis=1)
        tmp1 = list (np.squeeze(np.asarray(summation)))
        if(not np.prod(np.isclose(tmp1, tmp2))):
            print("Probabilities entered for emission matrix are invalid")

        summation = np.sum(trans_prob,axis=1)
        tmp1 = list (np.squeeze(np.asarray(summation)))

        if(not np.prod(np.isclose(tmp1, tmp2))):
            print("Probabilities entered for transition matrix are invalid")

        summation = np.sum(start_prob,axis=1)
        if (not np.isclose(summation[0,0], 1)):
            print("Probabilities entered for start state are invalid")

    def generate_state_map(self):
        self.state_map = {}
        for i,o in enumerate(self.states):
            self.state_map[i] = o

    def generate_obs_map(self):
        self.obs_map = {}
        for i,o in enumerate(self.observations):
            self.obs_map[o] = i
    #Forward prob test
    def forward(self,observations):
        total_stages = len(observations)

        ob_ind = self.obs_map[ observations[0] ]
        alpha = np.multiply ( np.transpose(self.em_prob[:,ob_ind]) , self.start_prob )

        for curr_t in range(1,total_stages):
            ob_ind = self.obs_map[observations[curr_t]]
            alpha = np.dot( alpha , self.trans_prob)
            alpha = np.multiply( alpha , np.transpose( self.em_prob[:,ob_ind] ))

        total_prob = alpha.sum()
        return ( total_prob )
    #viterbi algorithm implemtationimplemtation
    def viterbi(self,observations):
        total_stages = len(observations)
        num_states = len(self.states)
        old_path = np.zeros( (total_stages, num_states) )
        new_path = np.zeros( (total_stages, num_states) )
        ob_ind = self.obs_map[ observations[0] ]
        delta = np.multiply ( np.transpose(self.em_prob[:,ob_ind]) , self.start_prob )
        delta = delta /np.sum(delta)
        old_path[0,:] = [i for i in range(num_states) ]
        for curr_t in range(1,total_stages):
            ob_ind = self.obs_map[ observations[curr_t] ]
            temp  =  np.multiply (np.multiply(delta , self.trans_prob.transpose()) , self.em_prob[:, ob_ind] )
            delta = temp.max(axis = 1).transpose()
            delta = delta /np.sum(delta)
            max_temp = temp.argmax(axis=1).transpose()
            max_temp = np.ravel(max_temp).tolist()
            for s in range(num_states):
                new_path[:curr_t,s] = old_path[0:curr_t, max_temp[s] ] 
            new_path[curr_t,:] = [i for i in range(num_states) ]
            old_path = new_path.copy()
        final_max = np.argmax(np.ravel(delta))
        best_path = old_path[:,final_max].tolist()
        best_path_map = [ self.state_map[i] for i in best_path]
        return best_path_map
    #Baum-Welch implementatinalgorithm implementatin
    def BW(self,observation_list, iterations, quantities):
        obs_size = len(observation_list)
        prob = float('inf')
        q = quantities
        for i in range(iterations):
            emProbNew = np.asmatrix(np.zeros((self.em_prob.shape)))
            transProbNew = np.asmatrix(np.zeros((self.trans_prob.shape)))
            startProbNew = np.asmatrix(np.zeros((self.start_prob.shape)))
            for j in range(obs_size):
                emProbNew= emProbNew + q[j] * self.train_emission(observation_list[j])
                transProbNew = transProbNew + q[j] * self.train_transition(observation_list[j])
                startProbNew = startProbNew + q[j] * self.train_start_prob(observation_list[j])
            em_norm = emProbNew.sum(axis = 1)
            trans_norm = transProbNew.sum(axis = 1)
            start_norm = startProbNew.sum(axis = 1)
            emProbNew = emProbNew/ em_norm
            startProbNew = startProbNew/ start_norm.transpose()
            transProbNew = transProbNew/ trans_norm.transpose()
            self.em_prob,self.trans_prob = emProbNew,transProbNew
            self.start_prob = startProbNew
            if prob -  self.log_prob(observation_list,quantities)>0.0000001:
                prob = self.log_prob(observation_list,quantities)
            else:
                return self.em_prob, self.trans_prob , self.start_prob
        return self.em_prob, self.trans_prob , self.start_prob
    #Tuning parameter 
    def alpha_cal(self,observations):
        num_states = self.em_prob.shape[0]
        total_stages = len(observations)
        ob_ind = self.obs_map[ observations[0] ]
        alpha = np.asmatrix(np.zeros((num_states,total_stages)))
        c_scale = np.asmatrix(np.zeros((total_stages,1)))
        alpha[:,0] = np.multiply ( np.transpose(self.em_prob[:,ob_ind]) , self.start_prob ).transpose()
        c_scale[0,0] = 1/np.sum(alpha[:,0])
        alpha[:,0] = alpha[:,0] * c_scale[0]
        for curr_t in range(1,total_stages):
            ob_ind = self.obs_map[observations[curr_t]]
            alpha[:,curr_t] = np.dot( alpha[:,curr_t-1].transpose() , self.trans_prob).transpose()
            alpha[:,curr_t] = np.multiply( alpha[:,curr_t].transpose() , np.transpose( self.em_prob[:,ob_ind] )).transpose()
            c_scale[curr_t] = 1/np.sum(alpha[:,curr_t])
            alpha[:,curr_t] = alpha[:,curr_t] * c_scale[curr_t]
        return (alpha,c_scale)

    def beta_cal(self,observations,c_scale):
        num_states = self.em_prob.shape[0]
        total_stages = len(observations)
        ob_ind = self.obs_map[ observations[total_stages-1] ]
        beta = np.asmatrix(np.zeros((num_states,total_stages)))
        beta[:,total_stages-1] = c_scale[total_stages-1]
        for curr_t in range(total_stages-1,0,-1):
            ob_ind = self.obs_map[observations[curr_t]]
            beta[:,curr_t-1] = np.multiply( beta[:,curr_t] , self.em_prob[:,ob_ind] )
            beta[:,curr_t-1] = np.dot( self.trans_prob, beta[:,curr_t-1] )
            beta[:,curr_t-1] = np.multiply( beta[:,curr_t-1] , c_scale[curr_t -1 ] )
        return beta

    def forward_backward(self,observations):
        num_states = self.em_prob.shape[0]
        num_obs = len(observations)
        alpha, c = self.alpha_cal(observations)
        beta = self.beta_cal(observations,c)
        prob_obs_seq = np.sum(alpha[:,num_obs-1])
        delta1 = np.multiply(alpha,beta)/ prob_obs_seq 
        delta1 = delta1/c.transpose()

        return delta1
    #Train emission for BW
    def train_emission(self,observations):
        new_em_prob = np.asmatrix(np.zeros(self.em_prob.shape))    
        selectCols=[]
        for i in range(self.em_prob.shape[1]):
            selectCols.append([])
        for i in range(len(observations)):
            selectCols[ self.obs_map[observations[i]] ].append(i)
        delta = self.forward_backward(observations)
        totalProb = np.sum(delta,axis=1)
        for i in range(self.em_prob.shape[0]):
            for j in range(self.em_prob.shape[1]):
                new_em_prob[i,j] = np.sum(delta[i,selectCols[j]])/totalProb[i]
        return new_em_prob
    #Train transition for BW
    def train_transition(self,observations):
        new_trans_prob = np.asmatrix(np.zeros(self.trans_prob.shape))
        alpha,c = self.alpha_cal(observations)
        beta = self.beta_cal(observations,c)

        for t in range(len(observations)-1):
            temp1 = np.multiply(alpha[:,t],beta[:,t+1].transpose())
            temp1 = np.multiply(self.trans_prob,temp1)
            new_trans_prob = new_trans_prob + np.multiply(temp1,self.em_prob[:,self.obs_map[observations[t+1]]].transpose())
        for i in range(self.trans_prob.shape[0]):
            new_trans_prob[i,:] = new_trans_prob[i,:]/np.sum(new_trans_prob[i,:])
        return new_trans_prob
    #Train start_prob for BW
    def train_start_prob(self,observations):
        delta = self.forward_backward(observations)
        norm = sum(delta[:,0])
        return delta[:,0].transpose()/norm
    #calculate log 
    def log_prob(self,observations_list, quantities): 
        prob = 0
        for q,obs in enumerate(observations_list):
            temp,c_scale = self.alpha_cal(obs)
            prob = prob +  -1 *  quantities[q] * np.sum(np.log(c_scale))
        return prob
