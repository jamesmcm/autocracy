CXX ?= g++

# Democracy 3 uses the pre-C++11 libstdc++ string ABI.  The probe must use the
# same ABI because it calls the game's std::string member-function boundaries.
CXXFLAGS ?= -std=c++17 -O0 -g -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 \
	-Wall -Wextra -Wpedantic
LDFLAGS ?= -shared

TARGET := libd3probe.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): native_probe.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
