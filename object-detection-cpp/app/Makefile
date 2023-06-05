PROG1	= object_detection
OBJS1	= $(PROG1).cpp argparse.cpp imgprovider.cpp imgutils.cpp
PROGS	= $(PROG1)

PKGS = gio-2.0 vdostream gio-unix-2.0 liblarod cairo

CXXFLAGS  += -Iinclude

LDLIBS  += -ljpeg -laxoverlay
LDFLAGS += -L./lib -Wl,-rpath,'$$ORIGIN/lib'

CXXFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))

CXXFLAGS += -Wall -DLAROD_API_VERSION_3

all:	$(PROGS)

$(PROG1): $(OBJS1)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f $(PROGS) *.o *.eap lib/* include/* model/* label/*
