#include "ale_c_wrapper.hpp"

#include <cstring>
#include <string>
#include <stdexcept>
#include <iostream>

void fillRgbFromPalette(uint8_t *rgb, const uint8_t *obs, size_t rgb_size,
                            size_t obs_size) {
  assert(obs_size >= 0);
  assert(rgb_size == 3 * obs_size);

  for (int index = 0ul; index < obs_size; ++index) {
    uint8_t r, g, b;
    ALEInterface::getRGB(obs[index], r, g, b);
/*
    rgb[r_offset + index] = r;
    rgb[g_offset + index] = g;
    rgb[b_offset + index] = b;
*/
    rgb[index] = r;
    rgb[index] = g; index++;
    rgb[index] = b; index++;
  }
}

ALEInterface *ALE_new(const char *rom_file) {
  return new ALEInterface(rom_file);
}

void ALE_del(ALEInterface *ale) { delete ale; }

double act(ALEInterface *ale, int action) {
  assert(action >= static_cast<int>(ale::PLAYER_A_NOOP) &&
         action <= static_cast<int>(ale::PLAYER_A_DOWNLEFTFIRE));
  return ale->act(static_cast<ale::Action>(action));
}

int getScreenWidth(const ALEInterface *ale) {
  return ale->getScreen().width();
}

int getScreenHeight(const ALEInterface *ale) {
  return ale->getScreen().height();
}

bool isGameOver(const ALEInterface *ale) { return ale->gameOver(); }

void resetGame(ALEInterface *ale) {
  ale->resetGame();
  assert(!ale->gameOver());
}

bool loadState(ALEInterface *ale) { return ale->loadState(); }

void saveState(ALEInterface *ale) { ale->saveState(); }

void fillObs(const ALEInterface *ale, uint8_t *obs, size_t obs_size) {
  const ale::ALEScreen& screen = ale->getScreen();
  size_t h = screen.height();
  size_t w = screen.width();
  assert(obs_size == h * w);

  std::copy(screen.getArray().begin(), screen.getArray().end(), obs);
}

void fillRamObs(const ALEInterface *ale, uint8_t *ram, size_t ram_size) {
	const ale::ALERAM& ale_ram = ale->getRAM();
	assert(ram_size == ale_ram.size());

	const unsigned char* ram_content = ale_ram.array();
	std::copy(ram_content, ram_content + ram_size, ram);
}

int numLegalActions(ALEInterface *ale) {
  return static_cast<int>(ale->getMinimalActionSet().size());
}

void legalActions(ALEInterface *ale, int *actions,
                      size_t actions_size) {
  const std::vector<enum ale::Action>& legal_actions = ale->getMinimalActionSet();
  assert(actions_size == legal_actions.size());
  std::copy(legal_actions.begin(), legal_actions.end(), actions);
}

int livesRemained(const ALEInterface *ale) { return ale->lives(); }

int getSnapshotLength(const ALEInterface *ale) {
  return static_cast<int>(ale->getSnapshot().size());
}

void saveSnapshot(const ALEInterface *ale, uint8_t *data, size_t length) {
  std::string result = ale->getSnapshot();

  assert(length >= result.size() && length > 0);

  if (length < result.size())
    data = NULL;
  else
    result.copy(reinterpret_cast<char *>(data), length);
}

void restoreSnapshot(ALEInterface *ale, const uint8_t *snapshot,
                         size_t size) {
  assert(size > 0);

  std::string snapshotStr(reinterpret_cast<char const *>(snapshot), size);
  ale->restoreSnapshot(snapshotStr);
}

int maxReward(const ALEInterface *ale){
  return ale->maxReward();
}

