# Artificial dataset

# generate a few item types
np.random.seed(42)
mean = np.zeros(5)
cov = 5.*np.eye(5)
num_item_types = 5
item_cores = np.random.multivariate_normal(mean, cov, item_types)

# generate a few user types
mean_ = np.zeros(5)
cov_ = 5.*np.eye(5)
num_user_types = 5
user_cores = np.random.multivariate_normal(mean_, cov_, user_types)

# generate 10 items for every item type
items = np.zeros((0,5))
item_cov = 1.*np.eye(5)
num_items_per_type = 5 
for item_core in item_cores:
    items = np.vstack((items, np.random.multivariate_normal(item_core, item_cov, num_items_per_type)))
    
# generate 10 items for every item type
users = np.zeros((0,5))
user_cov = 1.*np.eye(5)
num_users_per_type = 5 
for user_core in user_cores:
    users = np.vstack((users, np.random.multivariate_normal(user_core, user_cov, num_users_per_type)))

# generate
ratings_precursor = np.dot(users, items.T)
t = 5
ratings_prob = 1/(1+np.exp(-ratings_precursor/t))
ratings = np.random.binomial(1,ratings_prob)





   