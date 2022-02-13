##ASDIV

#Sandra took six cups of coffee and Marcie took two cups of coffee. print the number of cups of coffee did Sandra and Marcie take in total?
cups_sandra = 6
cups_marcie = 2
total_cups = cups_sandra + cups_marcie
print(total_cups)

#Mrs. Franklin had 58 Valentines. Mrs. Franklin gave some to her students. Now she has 16. Write a program that prints how many Valentines did Mrs. Franklin give to her students?
valentines_franklin = 58
valentines_students = 16
total_valentines = valentines_franklin - valentines_students
print(total_valentines)

#Susie's father repaired the bookshelves in the reading room. If he has 210 books to be distributed equally on the 10 shelves he repaired, write a program that prints how many books will each shelf contain?
books_susie = 210
shelves_susie = 10
books_per_shelf = books_susie / shelves_susie
print(books_per_shelf)

#Michelle likes to save money every now and then so that she has money to buy the things that she wants. One day, she decided to count her savings. She opened her piggy bank and sorted out the different coins and dollar bills. If she counted a total of 20 nickels (a nickel is equivalent to 5 cents), write a program that prints what is the total value of money does she have in nickels?
nickels = 20
nickel_value = 5
total_nickels = nickels * nickel_value
print(total_nickels)

#Ducks nned to eat 3.5 pounds of insects each week to survive. If there is flock of ten ducks, write a program that prints how many pounds of insects do they need per day
flock_ducks = 10
pound_insects = 3.5
insects_per_day = (flock_ducks * pound_insects) / 7
print(insects_per_day)


##GSM8k - Two different code suggestions for the same questions that get different parts of the answer wrong.
##For example here the first code suggestion is missing the fraction of the drinks that is supposed to be water and the second one is missing the spill. 

#I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. But as I pour it, I spill one liter of the orange drink. Write a program that prints how much water is  in the remaining 24 liters

liters_orange = 10
liters_pineapple = 15
liters_orange_spill = 1
liters_pineapple_spill = 0
liters_pineapple_remaining = (liters_pineapple - liters_pineapple_spill)
liters_orange_remaining = (liters_orange - liters_orange_spill)
liters_orange_total = liters_orange_remaining + liters_pineapple_remaining
print(liters_orange_total)

liters_orange = 10
liters_pineapple = 15
liters_orange_water = liters_orange * 0.67
liters_pineapple_water = liters_pineapple * 0.75
liters_remaining = liters_orange_water + liters_pineapple_water
print(liters_remaining)