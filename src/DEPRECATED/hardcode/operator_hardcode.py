#  DD_a1m = {
#     # D-meson operators
        
#     "D[000]": QuantumNum(name='D_000',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_0'),
#     "D[001]": QuantumNum(name='D_001',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_1'),
#     "D[011]": QuantumNum(name='D_011',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_1_1'),
#     "D[002]": QuantumNum(name='D_011',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_2'),
#     "D[111]": QuantumNum(name='D_001',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_1_1_1')}

#     # D*-meson operators
# meep = {
#     "D*[000]": QuantumNum(name='D_000',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_0'),
#     "D*[001]": QuantumNum(name='D_001',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_1'),
#     "D*[011]": QuantumNum(name='D_011',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_1_1'),
#     "D*[002]": QuantumNum(name='D_011',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_0_0_2'),
#     "D*[111]": QuantumNum(name='D_001',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None,mom='mom_1_1_1'),

#     # D-meson operators w/ 1st order derivative 
#     "D[000]_nabla": QuantumNum(name='D_000',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv="nabla",mom='mom_0_0_0'),
#     "D[000]_a1xnabla": QuantumNum(name='D_000',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=True,deriv='nabla',mom='mom_0_0_0'),
#     "D[000]_b1xnabla": QuantumNum(name='D_000',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla",mom='mom_0_0_0'),
#     "D[000]_b1xnabla": QuantumNum(name='D_000',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla",mom='mom_0_0_0'),






#     "D_2": QuantumNum(name='D_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
#     "D_rhoxB": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=True,deriv="B"),
#     "D_nabla": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5],gamma_i=True,deriv="nabla"),
#     "D_Bxnabla": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=True,deriv="nabla"),


#     "D_rho2xB": QuantumNum(name='pion_2',had=1, F="A1", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=True,deriv="B"),

#     "D*[000]": QuantumNum(name='pion_2',had=1, F="T1m", flavor='charm,light',twoI=0, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv=None,mom='mom_0_0_0'),
#     "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B")
# }

# DD_t1p = {
#     "pion_2": QuantumNum(name='pion_2',had=1, F="A1", flavor='light',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
#     "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B")

# }


# a1_p_dpi_nomix = {
#     "pion_2": QuantumNum(name='pion_2',had=1, F="A1", flavor='light',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
#     "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B"),
#    }



# a1_mp_nomix_charm = {
#     "D_2": QuantumNum(name='D_2',had=1, F="A1", flavor='charm',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
#     "pion": QuantumNum(name='pion',had=1, F="A1",flavor='light',twoI=1,S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None),
# }

# ''' construct interpolators for 4 J^PC values '''
# # this is test case see charmonium spectrum paper by hadspec 
# #  
# # kaon_single = {
# #     "kaon": QuantumNum(name="kaon",had=1, F="A1", twoI=1,strange=-1, S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False),
# #     # "kaon_2": QuantumNum(name="kaon_2",had=1, F="A1", twoI=1, strange=-1,S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None),
# #     # "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1", strange=-1,twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla"),
# # }

# ## MAYBE DONT MIX TIME REVERSAL ## 
# ## EXCLUDE G5 
# ## gevp no longer hyperbolic cos behavior so cant mix with periodic bdy conditions

# a1_mp_nomix = {
#     "pion_2": QuantumNum(name='pion_2',had=1, F="A1", flavor='light',twoI=1, S=0, P=-1, C=1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
#     "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B"),
#    }
# a1_mp = {
#     "pion": QuantumNum(name='pion',had=1, F="A1",flavor='light',twoI=1,S=0, P=-1, C=1, gamma=gamma[5],gamma_i=False,deriv=None),
#     "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1",flavor='light', twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla"),
#     "rhoxB_A1": QuantumNum(name='rhoxB_A1',had=1, F="A1",flavor='light',twoI=1, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv="B"),
# }
# a1_mp_strange_nomix = {
#     "kaon_2": QuantumNum(name="kaon_2",had=1, F="A1", twoI=1, flavor='strange',S=1, P=-1, C=None, gamma=gamma[4]@gamma[5],gamma_i=False,deriv=None),
#     "rho_2xB_A1": QuantumNum(name='rho_2xB_A1',had=1, F="A1",flavor='strange',twoI=1, S=0, P=-1, C=1, gamma=gamma[4],gamma_i=True,deriv="B"),
# }
# a1_mp_strange = {
#     "kaon": QuantumNum(name="kaon",had=1, F="A1", twoI=1,flavor='strange', S=0, P=-1, C=None, gamma=gamma[5],deriv=None,gamma_i=False),
#     "b_1xNABLA_A1": QuantumNum(name='b_1xB_A1',had=1, F="A1", flavor='strange',twoI=1, S=0, P=-1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="nabla"),
#     "rhoxB_A1": QuantumNum(name='rhoxB_A1',had=1, F="A1", flavor='strange',twoI=1, S=0, P=-1, C=1, gamma=IDEN,gamma_i=True,deriv="B"),
# }


# # }

# # light_isovector = {
# #      "rho":  QuantumNum(name='rho',had=1, F="T1", twoI=1, strange=0,S=0, P=-1, C=-1, gamma=IDEN,gamma_i=True,deriv=None),
# #     "rho_2":  QuantumNum(name='rho_2',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv=None),
# #     "pion": QuantumNum(name='pion',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=False,deriv=None),
# #     "pion_2": QuantumNum(name='pion_2',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5]@gamma[4],gamma_i=False,deriv=None),
# # }


# # j_m_m = {
# #     'a1xNABLA_A1': QuantumNum(name='a1xNABLA_A1',had=1, F="A1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=True,deriv="nabla"),
# #     "rho":  QuantumNum(name='rho',had=1, F="T1", twoI=1, strange=0,S=0, P=-1, C=-1, gamma=I,gamma_i=True,deriv=None),
# #     "rho_2":  QuantumNum(name='rho_2',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv=None),
# #     "a0xNABLA":  QuantumNum(name='a0xNABLA',had=1, F="T1", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=I,gamma_i=False,deriv="nabla"),
# #     "a1xNABLA_T2":  QuantumNum(name='a1xNABLA_T2',had=1, F="T2", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[5],gamma_i=False,deriv="nabla"),
# #     "rhoxD_A2":  QuantumNum(name='rhoxD_A2',had=1, F="A2", twoI=1,strange=0, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv="D"),
# #     # "rhoxD_A2":  QuantumNum(name='rhoxD_A2',had=1, F="A2", twoI=1, S=0, P=-1, C=-1, gamma=gamma[4],gamma_i=True,deriv="D"),

# # }

# # # j_p_p = {
# # #     "a0": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=False,deriv=None),
# # #     "a1": QuantumNum(had=1, F="T1", twoI=1, S=0, P=1, C=1, gamma=gamma[5],gamma_i=True,deriv=None),
# # #     "rhoxNABLA_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=True,deriv="nabla"),
# # #     "rhoxNABLA_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=I,gamma_i=True,deriv="nabla"),
# # #     "b_1xB_A1": QuantumNum(had=1, F="A1", twoI=1, S=0, P=1, C=1, gamma=gamma[4]@gamma[5],gamma_i=True,deriv="B"),

# # # }

# # j_p_m = {


# # }


# @dataclass
# class ProjectedOperator:
#     dir: str
#     irrep: str # A1,T1
#     mom: str #tuple of ints without commas "000" "001" etc 
#     t0: int 
#     tz: int
#     states: list[int]