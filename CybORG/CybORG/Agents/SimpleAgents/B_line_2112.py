from CybORG.Agents import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices, Sleep


class B_lineAgent(BaseAgent):
    def __init__(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}
        self.jumps = [0,1,2,2,2,2,5,5,5,5,9,9,9,12,13]

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass

    def get_action(self, observation, action_space):
        # print(self.action)
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        session = 0

        while True:
            if observation['success'] == True:
                self.action += 1 if self.action < 14 else 0
            else:
                self.action = self.jumps[self.action]

            if self.action in self.action_history:
                action = self.action_history[self.action]

            # Discover Remote Systems
            elif self.action == 0:
                self.last_subnet = observation['User0']['Interface'][0]['Subnet']
                action = DiscoverRemoteSystems(session=session, agent='Red', subnet=self.last_subnet)
                print(action)
            # Discover Network Services- new IP address found
            elif self.action == 1:
                self.last_ip_address = [value for key, value in observation.items() if key != 'success'][1]['Interface'][0]['IP Address']
                action =DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)
                print(action)
            # Exploit User1
            elif self.action == 2:
                 action = ExploitRemoteService(session=session, agent='Red', ip_address=self.last_ip_address)
                 print(action)
            # Privilege escalation on User1
            elif self.action == 3:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)
                print(action)
            # Discover Network Services- new IP address found
            elif self.action == 4:
                self.last_ip_address = observation['Enterprise1']['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)
                print(action)
            # Exploit- Enterprise1
            elif self.action == 5:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)
                print(action)
            # Privilege escalation on Enterprise1
            elif self.action == 6:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)
                print(action)
            # Scanning the new subnet found.
            elif self.action == 7:
                self.last_subnet = observation['Enterprise1']['Interface'][0]['Subnet']
                action = DiscoverRemoteSystems(subnet=self.last_subnet, agent='Red', session=session)
                print(action)
            # Discover Network Services- Enterprise2
            elif self.action == 8:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][2]['Interface'][0]['IP Address']
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=self.target_ip_address)
                print(action)
            # Exploit- Enterprise2
            elif self.action == 9:
                self.target_ip_address = [value for key, value in observation.items() if key != 'success'][0]['Interface'][0]['IP Address']
                action = ExploitRemoteService(session=session, agent='Red', ip_address=self.target_ip_address)
                print(action)
            # Privilege escalation on Enterprise2
            elif self.action == 10:
                hostname = [value for key, value in observation.items() if key != 'success' and 'System info' in value][0]['System info']['Hostname']
                action = PrivilegeEscalate(agent='Red', hostname=hostname, session=session)
                print(action)
            # Discover Network Services- Op_Server0
            elif self.action == 11:
                action = DiscoverNetworkServices(session=session, agent='Red', ip_address=observation['Op_Server0']['Interface'][0]['IP Address'])
                print(action)
            # Exploit- Op_Server0
            elif self.action == 12:
                info = [value for key, value in observation.items() if key != 'success']
                if len(info) > 0:
                    action = ExploitRemoteService(agent='Red', session=session, ip_address=info[0]['Interface'][0]['IP Address'])
                    print(action)
                else:
                    self.action = 0
                    print(action)
                    continue
                
            # Privilege escalation on Op_Server0
            elif self.action == 13:
                action = PrivilegeEscalate(agent='Red', hostname='Op_Server0', session=session)
            # Impact on Op_server0
            elif self.action == 14:
                action = Impact(agent='Red', session=session, hostname='Op_Server0')

            if self.action not in self.action_history:
                self.action_history[self.action] = action
            return action

    def end_episode(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}

    def set_initial_values(self, action_space, observation):
        pass