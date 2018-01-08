#ifndef __ALE_C_WRAPPER_H__
#define __ALE_C_WRAPPER_H__

#include <stdexcept>
#include <cassert>
#include <algorithm>

#include "xitari/ale_interface.hpp"

typedef unsigned char uint8_t;

typedef ale::ALEInterface ALEInterface;


extern "C" {
    void getScreenRGB(ALEInterface *ale, uint8_t *rgb, size_t rgb_size) {
      //std::cout << "getScreenRGB" << std::endl;
        
      const ale::ALEScreen& screen = ale->getScreen();
      size_t h = screen.height();
      size_t w = screen.width();
      size_t obs_size = h * w;
      const std::vector<unsigned char>& obs = screen.getArray();

      uint8_t* p = rgb;
      const unsigned char* op = obs.data();
        
      for (int index = 0ul; index < obs_size; ++index) {
        uint8_t r, g, b;
        ALEInterface::getRGB(*op, r, g, b); op++;
    /*
        rgb[r_offset + index] = r;
        rgb[g_offset + index] = g;
        rgb[b_offset + index] = b;
    */
        *p = r; p++;
        *p = g; p++;
        *p = b; p++;
      }
    }
    
	void fillRgbFromPalette(uint8_t *rgb, const uint8_t *obs, size_t rgb_size, size_t obs_size);

	ALEInterface *ALE_new(const char *rom_file);

	void ALE_del(ALEInterface *ale) ;

	double act(ALEInterface *ale, int action);

	int getScreenWidth(const ALEInterface *ale);

	int getScreenHeight(const ALEInterface *ale);

	bool isGameOver(const ALEInterface *ale);

	void resetGame(ALEInterface *ale);

	bool loadState(ALEInterface *ale);

	void saveState(ALEInterface *ale);

	void fillObs(const ALEInterface *ale, uint8_t *obs, size_t obs_size);

	void fillRamObs(const ALEInterface *ale, uint8_t *ram, size_t ram_size) ;

	int numLegalActions(ALEInterface *ale);
	
	void legalActions(ALEInterface *ale, int *actions, size_t actions_size);

	int livesRemained(const ALEInterface *ale);

	int getSnapshotLength(const ALEInterface *ale);
	void saveSnapshot(const ALEInterface *ale, uint8_t *data, size_t length);

	void restoreSnapshot(ALEInterface *ale, const uint8_t *snapshot, size_t size);

    int maxReward(const ALEInterface *ale);
}

#endif
