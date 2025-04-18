import unittest, os, re, sys
import unittest.mock as mock
from unittest.mock import patch, call
from factor_model_optimizer import *
import yaml

class TestBase(unittest.TestCase):

    def setUp(self) -> None:
         self.betas = np.random.rand(8,2)
         self.C = np.diag((1,2))
         self.A = np.diag([0.2]*4 + [0.25]*4)
         self.sigma = self.betas@self.C@self.betas.T + self.A
    #     self.rextests = TestsManager(re.search)
    #     self.nametests = TestsManager(lambda name1,name2: name1 == name2)
    #     self.filenametests = TestsManager(lambda name1,name2: os.path.splitext(name1)[0] == os.path.splitext(name2)[0])
    #     self.extensiontests = TestsManager(lambda extension, name: os.path.splitext(name)[1] == extension)
    #     self.stopflag = 'stop'
    #     self.root = r'G:\mockroot'
    #     self.dirs = r'mockdir mockotherdir'.split()
    #     self.stoppath = os.path.join(self.root,self.dirs[0],self.stopflag)

    #     self.rextest = self.rextests.curry('a b'.split())
    #     self.nametest = self.filenametests.curry('cxlever.odt'.split())
    #     self.extensiontest = self.extensiontests.curry('.env .sys'.split())
    #     self.tests = [partial(no_positives,test) for test in (self.rextest,self.nametest,self.extensiontest)]

    #     self.is_stopped = partial(has_file,self.stopflag)

# class TestFileSelection(TestBase):
       
#        def test_combined_selection_only1(self):
#             self.assertEqual(cumulative_tests(self.speedfiles,self.tests),'clever.xls clumsy.odt'.split())

#        def test_combined_selection_only2(self):
#             self.tests[1] = partial(no_positives,self.filenametests.curry('clever.odt'.split()))
#             self.assertEqual(cumulative_tests(self.speedfiles,self.tests),'clumsy.odt'.split())
# 

class TestTest(TestBase):
    def test_dummy_test(self):
        print('betas\n',self.betas)
        self.assertEqual(self.betas.shape,(8,2))

class TestWood(TestBase):

    def test_inversion(self):
        sigma = self.betas@self.C@self.betas.T + self.A
        # print(sigma)
        self.assertTrue(np.allclose(self.sigma@Factor_Model.woodbury_fm(self.A,self.betas,self.C),np.eye(len(sigma))))
        

def suite():
    # pass
    print('running suite')
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWood))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTest))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFileSelection))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDirectorySelection))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStopflag))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWalk))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUpdateInitialization))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPathMethodsOnGdrive))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRexs))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExtensions))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNames))
    # suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStopflag))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

    # unittest.main()
