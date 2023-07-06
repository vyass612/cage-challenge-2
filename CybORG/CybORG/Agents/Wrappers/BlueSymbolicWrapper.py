from gym import Env
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper, ChallengeWrapper
##get the output of challenge wrapper in vector form. convert this to a symbolic format shown below

#model.add_data({
#    OpServerCompromise: {
#        ('Op_Server0 Scan'): Fact.FALSE,
#        ('Op_Server0 Exploit'): Fact.FALSE,
#        ('Op_Server0 Impact'): Fact.TRUE,
#        ('Op_Server0 Privilege Escalation'): Fact.FALSE},
#    
#    Bias: {
#        ( 'Bias Operand'): Fact.TRUE,
#    }
#})

class BlueSymbolicWrapper(Env,BaseWrapper):