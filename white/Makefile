CXX := g++
CXXFLAGS := -O3 -std=c++17 -pthread -Wall -Wextra

all: dirhash_fast dirhash_fast_mt

dirhash_fast: dirhash_fast.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

dirhash_fast_mt: dirhash_fast_mt.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Optional: replace dirhash_fast with the MT build
fast: dirhash_fast_mt
	cp -f dirhash_fast_mt dirhash_fast

clean:
	rm -f dirhash_fast dirhash_fast_mt

.PHONY: all clean fast

