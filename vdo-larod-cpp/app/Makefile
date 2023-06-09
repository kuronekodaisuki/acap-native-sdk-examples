PROG1	= vdo_larod
OBJS1	= $(PROG1).cpp imgprovider.cpp
PROGS	= $(PROG1)

PKGS = gio-2.0 vdostream gio-unix-2.0 liblarod cairo

CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))
LDLIBS += -lm -laxoverlay

CFLAGS += -Wall
# This example is based on larod version 3
CFLAGS += -DLAROD_API_VERSION_3

all: $(PROGS)

$(PROG1): $(OBJS1)
	$(CXX) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f $(PROGS) *.o *.eap
