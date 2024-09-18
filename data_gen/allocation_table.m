function allocation_index = allocation_table(users)
%This function generates the allocation index table for 802.11ax for one 20
%MHz channel in 802.11ax

user_compare = [9, 8, 7, 6, 5, 4, 3, 2, 1];
allocation_index_table = [0, 8, 9, 11, 15, 112, 128, 96, 192];   

[row,col] = find(user_compare == users);
allocation_index = allocation_index_table(col); 

end