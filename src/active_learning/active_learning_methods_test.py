import unittest
import tensorflow as tf
from active_learning_methods import diversity_sampling
from active_learning_utils import pivot_reorder



class TestMethods(unittest.TestCase):
    ''' Unit tests for some methods in active_learning_methods.py and in active_learning_utils.py '''
    
    def test_diversity(self):
        img     = tf.zeros([224, 224, 3])
        label   = tf.constant(1, dtype=tf.int64)
        images  = tf.data.Dataset.from_tensor_slices([img, img, img, img])
        labels  = tf.data.Dataset.from_tensor_slices([label, label, label, label])
        
        prio_order  = [0,1,2,3]
        unseen_data = tf.data.Dataset.zip((images, labels))
        
        #   [(class_name, class_description, score, class_id)]
        y_pred = [
            [("person_id",  "person",       0.9,       0)],
            [("car_id",     "car"   ,       0.8,       1)],
            [("car_id",     "car"   ,       0.8,       1)],
            [("truck_id",   "truck" ,       0.7,       2)],
        ]
        
        prio = diversity_sampling(prio_order, unseen_data, y_pred)
        self.assertEqual(prio, [0,1,3,2], f"Expected [0,1,3,2], got {prio}")
    
    def test_diversity_budget(self):
        img     = tf.zeros([224, 224, 3])
        label   = tf.constant(1, dtype=tf.int64)
        images  = tf.data.Dataset.from_tensor_slices([img, img, img, img])
        labels  = tf.data.Dataset.from_tensor_slices([label, label, label, label])
        
        prio_order  = [0,1,2,3,4,5,6,7]
        unseen_data = tf.data.Dataset.zip((images, labels))
        
        #   [(class_name, class_description, score, class_id)]
        y_pred = [
            [("person_id",  "person",       0.9,       0)],
            [("car_id",     "car"   ,       0.8,       1)],
            [("car_id",     "car"   ,       0.8,       1)],
            [("truck_id",   "truck" ,       0.7,       2)],
            [("truck_id",   "truck" ,       0.6,       2)],
            [("truck_id",   "truck" ,       0.9,       2)],
            [("truck_id",   "car"   ,       0.5,       1)],
            [("truck_id",   "car"   ,       0.5,       1)],
        ]
        
        prio = diversity_sampling(prio_order, unseen_data, y_pred)
        self.assertEqual(prio, [0,1,3,2,4,5,6,7], f"Expected [0,1,3,2,4,6,5,7], got {prio}")
    
    def test_diversity_budget_reverse(self):
        img     = tf.zeros([224, 224, 3])
        label   = tf.constant(1, dtype=tf.int64)
        images  = tf.data.Dataset.from_tensor_slices([img, img, img, img])
        labels  = tf.data.Dataset.from_tensor_slices([label, label, label, label])
        
        prio_order  = list(reversed([0,1,2,3,4,5,6,7]))
        unseen_data = tf.data.Dataset.zip((images, labels))
        
        #   [(class_name, class_description, score, class_id)]
        y_pred = [
            [("person_id",  "person",       0.9,       0)],
            [("car_id",     "car"   ,       0.8,       1)],
            [("car_id",     "car"   ,       0.8,       1)],
            [("truck_id",   "truck" ,       0.7,       2)],
            [("truck_id",   "truck" ,       0.6,       2)],
            [("truck_id",   "truck" ,       0.9,       2)],
            [("truck_id",   "car"   ,       0.5,       1)],
            [("truck_id",   "car"   ,       0.5,       1)],
        ]
        
        prio = diversity_sampling(prio_order, unseen_data, y_pred)
        self.assertEqual(prio, [7,5,0,6,4,3,2,1], f"Expected [7,5,0,6,4,3,2,1], got {prio}")

    def test_pivot_reorder_middle(self):
        a = [0,1,2,3,4,5,6,7,8]
        budget = 3
        
    def test_pivot_reorder_uneven(self):
        a = [0,1,2,3,4,5,6,7,8]
        budget = 3
        
        b = pivot_reorder(a, budget, 0)
        c = [0,1,2,3,4,5,6,7,8]
        self.assertEqual(b, c, f"Expected {c}, got {b}")
        
        f = pivot_reorder(a, budget, 0.5)
        g = [3,4,5,0,1,2,6,7,8]
        self.assertEqual(f, g, f"Expected {f}, got {g}")
        
        d = pivot_reorder(a, budget, 1)
        e = [6,7,8,0,1,2,3,4,5]
        self.assertEqual(d, e, f"Expected {d}, got {e}")
    
    def test_pivot_reorder_even(self):
        a = [0,1,2,3,4,5,6,7,8]
        budget = 4
        
        b = pivot_reorder(a, budget, 0)
        c = [0,1,2,3,4,5,6,7,8]
        self.assertEqual(b, c)
        
        f = pivot_reorder(a, budget, 0.5)
        g = [2,3,4,5,6,0,1,7,8]
        self.assertEqual(f, g)
        
        d = pivot_reorder(a, budget, 1)
        e = [4,5,6,7,8,0,1,2,3]
        self.assertEqual(d, e)
    
    def test_pivot_reorder_uneven_uneven(self):
        a = [0,1,2,3,4,5,6,7]
        budget = 3
        
        b = pivot_reorder(a, budget, 0)
        c = [0,1,2,3,4,5,6,7]
        self.assertEqual(b, c)
        
        f = pivot_reorder(a, budget, 0.5)
        g = [3,4,5,0,1,2,6,7]
        self.assertEqual(f, g)
        
        d = pivot_reorder(a, budget, 1)
        e = [5,6,7,0,1,2,3,4]
        self.assertEqual(d, e)
    
    def test_pivot_reorder_even_uneven(self):
        a = [0,1,2,3,4,5,6,7]
        budget = 4
        
        b = pivot_reorder(a, budget, 0)
        c = [0,1,2,3,4,5,6,7]
        self.assertEqual(b, c)
        
        f = pivot_reorder(a, budget, 0.5)
        g = [2,3,4,5,6,0,1,7]
        self.assertEqual(f, g)
        
        d = pivot_reorder(a, budget, 1)
        e = [3,4,5,6,7,0,1,2]
        self.assertEqual(d, e)

    def test_pivot_reorder_(self):
        a = [0,1,2,3,4,5,6,7]
        budget = 8
        
        b = pivot_reorder(a, budget, 0)
        c = [0,1,2,3,4,5,6,7]
        self.assertEqual(b, c)
        
if __name__ == '__main__':
    unittest.main()