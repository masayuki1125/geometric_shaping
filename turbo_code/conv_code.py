import numpy as np
import math 

class conv_code():
  INF = 1e6

  def __init__(self, back_poly, fwd_polys):

    # Number of bits in encoder state.
    self.mem_len = math.floor(math.log(back_poly) / math.log(2))

    # Encoder state space (integers in the range [0, 2 ** mem_len)).
    self.state_space = tuple(n for n in range(1 << self.mem_len))

    # Number of encoder output bits per input bit.
    self.n_out = len(fwd_polys)

    # MSB of next encoder state, given current state and input bit.
    self.next_state_msb = tuple(tuple(self.bitxor(back_poly & ((b << self.mem_len) + s))for s in self.state_space) for b in (0, 1))

    # Encoder output bits, given current state and input bit.
    self.out_bits = tuple(tuple(tuple(self.bitxor(p & ((self.next_state_msb[b][s] << self.mem_len) + s))for p in fwd_polys) for s in self.state_space) for b in (0, 1))
    
    # Next encoder state, given current state and input bit.
    self.next_state = tuple(tuple((self.next_state_msb[b][s] << (self.mem_len - 1)) + (s >> 1)for s in self.state_space) for b in (0, 1))

    return


# In[106]:


class conv_code(conv_code):

  @staticmethod
  def bitxor(num):
    '''
    Returns the XOR of the bits in the binary representation of the
    nonnegative integer num.
    '''

    count_of_ones = 0
    while num > 0:
      count_of_ones += num & 1
      num >>= 1

    return count_of_ones % 2

  @staticmethod
  def maxstar(eggs, spam, max_log=True):
    '''
    Returns log(exp(eggs) + exp(spam)) if not max_log, and max(eggs, spam)
    otherwise.
    '''
    return max(eggs, spam) + (0 if max_log else math.log(1 + math.exp(-abs(spam - eggs))))


# In[107]:


class conv_code(conv_code):

  def encode(self, info_bits):

    #info_bits = np.asarray(info_bits).ravel()
    n_info_bits = info_bits.size

    code_bits, enc_state = -np.ones(self.n_out * (n_info_bits + self.mem_len), dtype=int), 0
    
    for k in range(n_info_bits + self.mem_len):
      in_bit = (info_bits[k] if k < n_info_bits else self.next_state_msb[0][enc_state])
      #print((self.out_bits[in_bit][enc_state]))
      code_bits[self.n_out * k : self.n_out * (k + 1)] = (self.out_bits[in_bit][enc_state])
      enc_state = self.next_state[in_bit][enc_state]

    return code_bits


# In[108]:


class conv_code(conv_code):

  def _branch_metrics(self, out_bit_llrs, pre_in_bit_llr=0):

    gamma_val = ([pre_in_bit_llr / 2 for s in self.state_space],[-pre_in_bit_llr / 2 for s in self.state_space])
    for enc_state in self.state_space:
      for bit0, bit1, val in zip(self.out_bits[0][enc_state],self.out_bits[1][enc_state],out_bit_llrs):
        gamma_val[0][enc_state] += val / 2 if bit0 == 0 else -val / 2
        gamma_val[1][enc_state] += val / 2 if bit1 == 0 else -val / 2

    return gamma_val

  def _update_path_metrics(self, out_bit_llrs, path_metrics, best_bit):

    gamma_val = self._branch_metrics(out_bit_llrs)

    pmn = path_metrics[:]
    for enc_state in self.state_space:
      cpm0 = gamma_val[0][enc_state] + pmn[self.next_state[0][enc_state]]
      cpm1 = gamma_val[1][enc_state] + pmn[self.next_state[1][enc_state]]
      path_metrics[enc_state], best_bit[enc_state] = ((cpm0, 0) if cpm0 >= cpm1 else (cpm1, 1))

    return


# In[109]:


class conv_code(conv_code):
  def decode_viterbi(self, code_bit_llrs):

    code_bit_llrs = np.asarray(code_bit_llrs).ravel()
    n_in_bits = int(code_bit_llrs.size / self.n_out)
    n_info_bits = n_in_bits - self.mem_len

    # Path metric for each state at time n_in_bits.
    path_metrics = [(0 if s == 0 else -conv_code.INF) for s in self.state_space]

    # Best input bit in each state at times 0 to n_in_bits - 1.
    best_bit = [[-1 for s in self.state_space] for k in range(n_in_bits)]

    # Start at time n_in_bits - 1 and work backward to time 0, finding
    # path metric and best input bit for each state at each time.
    for k in range(n_in_bits - 1, -1, -1):
      self._update_path_metrics(
      code_bit_llrs[self.n_out * k : self.n_out * (k + 1)],path_metrics, best_bit[k])

    # Decode by starting in state 0 at time 0 and tracing path
    # corresponding to best input bits.
    info_bits_hat, enc_state = -np.ones(n_info_bits, dtype=int), 0
    for k in range(n_info_bits):
      info_bits_hat[k] = best_bit[k][enc_state]
      enc_state = self.next_state[info_bits_hat[k]][enc_state]

    return info_bits_hat

  def _update_alpha(self,out_bit_llrs,pre_in_bit_llr,alpha_val,alpha_val_next,max_log):

    gamma_val = self._branch_metrics(out_bit_llrs, pre_in_bit_llr)

    for enc_state in self.state_space:
      alpha_val_next[self.next_state[0][enc_state]] = self.maxstar(alpha_val_next[self.next_state[0][enc_state]],alpha_val[enc_state] + gamma_val[0][enc_state],max_log)
      alpha_val_next[self.next_state[1][enc_state]] = self.maxstar(alpha_val_next[self.next_state[1][enc_state]],alpha_val[enc_state] + gamma_val[1][enc_state],max_log)

    return


# In[110]:


class conv_code(conv_code):

  def _update_beta_tail(self, out_bit_llrs, beta_val, max_log):

    gamma_val = self._branch_metrics(out_bit_llrs, 0)

    bvn = beta_val[:]
    for enc_state in self.state_space:
      beta_val[enc_state] = self.maxstar(gamma_val[0][enc_state] + bvn[self.next_state[0][enc_state]],gamma_val[1][enc_state] + bvn[self.next_state[1][enc_state]],max_log)
    return

  def _update_beta(self,out_bit_llrs,pre_in_bit_llr,alpha_val,beta_val,max_log):

    gamma_val = self._branch_metrics(out_bit_llrs, pre_in_bit_llr)

    met0 = -conv_code.INF
    met1 = -conv_code.INF
    bvn = beta_val[:]
    for enc_state in self.state_space:
      beta_val[enc_state] = self.maxstar(gamma_val[0][enc_state] + bvn[self.next_state[0][enc_state]],gamma_val[1][enc_state] + bvn[self.next_state[1][enc_state]],max_log)
      met0 = self.maxstar(alpha_val[enc_state] + gamma_val[0][enc_state]+ bvn[self.next_state[0][enc_state]],met0,max_log)
      met1 = self.maxstar(alpha_val[enc_state] + gamma_val[1][enc_state]+ bvn[self.next_state[1][enc_state]],met1,max_log)
    return met0 - met1


# In[111]:


class conv_code(conv_code):
  def decode_bcjr(self,code_bit_llrs,pre_info_bit_llrs=None,max_log=True):

    code_bit_llrs = np.asarray(code_bit_llrs).ravel()
    n_in_bits = int(code_bit_llrs.size / self.n_out)
    n_info_bits = n_in_bits - self.mem_len

    if pre_info_bit_llrs is None:
      pre_info_bit_llrs = np.zeros(n_info_bits)
    else:
      pre_info_bit_llrs = np.asarray(pre_info_bit_llrs).ravel()

    # FORWARD PASS: Recursively compute alpha values for all states at
    # all times from 1 to n_info_bits - 1, working forward from time 0.
    alpha = [[(0 if s == 0 and k == 0 else -conv_code.INF)for s in self.state_space] for k in range(n_info_bits)]
    for k in range(n_info_bits - 1):
      out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
      self._update_alpha(out_bit_llrs, pre_info_bit_llrs[k],alpha[k], alpha[k + 1], max_log)

    # BACKWARD PASS (TAIL): Recursively compute beta values for all
    # states at time n_info_bits, working backward from time n_in_bits.
    beta = [(0 if s == 0 else -conv_code.INF) for s in self.state_space]
    for k in range(n_in_bits - 1, n_info_bits - 1, -1):
      out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
      self._update_beta_tail(out_bit_llrs, beta, max_log)

    # BACKWARD PASS: Recursively compute beta values for all states at
    # each time k from 0 to n_info_bits - 1, working backward from time
    # n_info_bits, and also obtaining the post-decoding LLR for the info
    # bit at each time.
    post_info_bit_llrs = np.zeros_like(pre_info_bit_llrs)
    for k in range(n_info_bits - 1, - 1, -1):
      out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
      post_info_bit_llrs[k] = self._update_beta(out_bit_llrs, pre_info_bit_llrs[k],alpha[k], beta, max_log)

    return post_info_bit_llrs