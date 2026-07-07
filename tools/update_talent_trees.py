import csv, io

path = r'data/sheet_exports/Talent Trees.csv'

with open(path, 'r', encoding='utf-8', newline='') as f:
    content = f.read()

rows = list(csv.reader(io.StringIO(content)))

modified = 0

for i, row in enumerate(rows):
    if len(row) < 8:
        continue
    uid = row[0].strip()
    talent = row[6].strip() if len(row) > 6 else ''
    desc = row[7] if len(row) > 7 else ''

    changed = False

    # TWILIGHT RANGER
    if uid == 'twilight_ranger':
        if talent == 'Cursed Magic':
            row[7] = desc + ' [Balance 5-16-26: attacks now deal 50% additional damage after claiming 50 Souls]'
            changed = True
        elif talent == 'Unity Magic':
            row[7] = desc.replace('increases by 200%', 'increases by 400% (was 200%)')
            changed = True
        elif talent == 'Critical Enchantment':
            row[7] = desc.replace('by 35%.', 'by 70% (was 35%).')
            changed = True
        elif talent == 'Crown of Darkness':
            row[7] = desc.replace('critical damage by 4%', 'critical damage by 5% (was 4%)')
            row[7] = row[7].replace('Magical Shot by 30%', 'Magical Shot by 50% (was 30%)')
            row[7] += ' [Balance 5-16-26: crit chance 4%->5%; shot damage 30%->50%]'
            changed = True

    # INQUISITOR
    elif uid == 'inquisitor':
        if talent == 'Shield of Faith':
            row[7] = desc.replace('restore her shield in 3.5 seconds',
                                  'restore her shield in 15 seconds (was 3.5s, raised to 15s in Balance 5-16-26)')
            changed = True
        elif talent == 'Ronin':
            row[7] = desc.replace('increases to 1100%.', 'increases to 800% (was 1100%).')
            row[7] = row[7].replace('[Patch 13.04: added +50% attack speed bonus when solo]',
                                    '[Patch 13.04: added +50% attack speed bonus when solo] [Balance 5-16-26: max limit 1100%->800%]')
            changed = True
        elif talent == 'Hammer of Faith':
            row[7] = desc + ' [Balance 5-16-26: stun now applies to monsters, bosses, and mini bosses (1 sec duration)]'
            changed = True

    # BARD
    elif uid == 'bard':
        if talent == 'Drummer':
            row[7] = desc.replace('increases by 2% per rank;', 'increases by 3% per rank (was 2%);')
            row[7] = row[7].replace('does not exceed 42%.', 'does not exceed 84% (was 42%).')
            row[7] += ' [Balance 5-16-26: damage 2%->3% per rank; limit 42%->84%]'
            changed = True
        elif talent == 'Cacophony':
            row[7] = desc.replace('inflict 100% of', 'inflict 150% of (was 100%)')
            row[7] += ' [Balance 5-16-26: clump damage 100%->150%]'
            changed = True

    # BRUISER
    elif uid == 'bruiser':
        if talent == 'On My Treat!':
            row[7] = desc.replace('with a 10% chance when collecting a Clover leaf,',
                                  'with a 20% chance when collecting a Clover leaf (was 10%),')
            row[7] = row[7].replace("reduces monsters' armor by 30% to a maximum of 60%",
                                    "reduces monsters' armor by 35% to a maximum of 70% (was 30%/60%)")
            row[7] = row[7].replace('The puddle exists for 10 sec.', 'The puddle exists for 15 sec (was 10s).')
            row[7] += ' [Balance 5-16-26: puddle 10s->15s; armor 30%->35% (max 60%->70%); leaf throw chance 10%->20%]'
            changed = True
        elif talent == 'No Worries!':
            row[7] = desc.replace('cleanses negative effects from 2 random Bruisers.',
                                  'cleanses negative effects from 3 random Bruisers (was 2). Also has a 20% chance to grant an additional Clover leaf upon receiving one.')
            row[7] += ' [Balance 5-16-26: cleanses 2->3; new 20% extra leaf mechanic]'
            changed = True

    # BANSHEE
    elif uid == 'banshee':
        if talent == 'Spirit of Vengeance':
            row[7] = desc.replace('increased by 10%.', 'increased by 15% (was 10%).')
            row[7] += ' [Balance 5-16-26: damage bonus 10%->15%]'
            changed = True
        elif talent == 'Infirmity':
            row[7] = desc.replace('reduced by 15%.', 'reduced by 20% (was 15%).')
            row[7] += ' [Balance 5-16-26: armor reduction 15%->20%]'
            changed = True

    # GENIE
    elif uid == 'genie':
        if talent == 'Spiritual Cleansing':
            row[7] = desc.replace('cleanses 3 Genies', 'cleanses 4 Genies (was 3)')
            row[7] += ' [Balance 5-16-26: cleansed units 3->4]'
            changed = True

    # CULTIST
    elif uid == 'cultist':
        if talent == 'Harbinger of Evil':
            row[7] = desc.replace('Each soul increases Cultist damage by 2%.',
                                  'Each soul increases Cultist damage by 2.5% (was 2%).')
            row[7] = row[7].replace('increases their damage AND attack speed by 15%.',
                                    'increases their damage AND attack speed by 30% (was 15%).')
            row[7] = row[7].replace('[Patch 13.04: Harbinger bonus now grants both damage and attack speed (was attack speed only)]',
                                    '[Patch 13.04: Harbinger bonus now grants both damage and attack speed (was attack speed only)] [Balance 5-16-26: soul bonus 2%->2.5%; Harbinger bonus 15%->30%]')
            changed = True
        elif talent == 'Harbinger of the Abyss':
            row[7] = desc.replace('increasing attack speed by 30%.', 'increasing attack speed by 35% (was 30%).')
            row[7] = row[7].replace('[Patch 13.04: completely reworked — Demon form threshold changed, soul mechanic overhauled]',
                                    '[Patch 13.04: completely reworked] [Balance 5-16-26: Harbinger attack speed 30%->35%]')
            changed = True
        elif talent == 'Unity of the Cult':
            row[7] = desc.replace('damage by 5% for every Cultist', 'damage by 7% for every Cultist (was 5%)')
            row[7] += ' [Balance 5-16-26: per-Cultist bonus 5%->7%]'
            changed = True

    # ROBOT
    elif uid == 'robot':
        if talent == 'Successful Tuning':
            row[7] = desc.replace('has a 40% chance', 'has a 60% chance (was 40%)')
            row[7] += ' [Balance 5-16-26: upgrade chance 40%->60%]'
            changed = True
        elif talent == 'Fire Support':
            row[7] = desc.replace('by 2.5% per Part,', 'by 3% per Part (was 2.5%),')
            row[7] = row[7].replace('maximum of 70%.', 'maximum of 90% (was 70%).')
            row[7] += ' [Balance 5-16-26: crit bonus 2.5%->3% per Part; cap 70%->90%]'
            changed = True
        elif talent == 'Speed Module':
            row[7] = desc.replace('by 30% for 20s.', 'by 35% for 20s (was 30%).')
            row[7] = row[7].replace('[Patch 13.04: reworked — was 40% attack speed only, now 30% damage+attack speed]',
                                    '[Patch 13.04: reworked — was 40% attack speed only, now damage+attack speed] [Balance 5-16-26: bonus 30%->35%]')
            changed = True
        elif talent == 'BRG 1000':
            row[7] = desc.replace('400% damage to common monsters', '650% damage to common monsters (was 400%)')
            row[7] = row[7].replace('800% damage to bosses and mini bosses.', '960% damage to bosses and mini bosses (was 800%).')
            row[7] += ' [Balance 5-16-26: monster dmg 400%->650%; boss dmg 800%->960%]'
            changed = True
        elif talent == 'General Program':
            row[7] = desc.replace('dealing 150% of the Robot\'s critical damage in the area.',
                                  'dealing 75% of the Robot\'s critical damage in the area (actual; game description erroneous), with a 1s cooldown between shots.')
            row[7] = row[7].replace('damage by 30%.', 'damage by 15% (was 30%).')
            row[7] = row[7].replace('[Patch 13.04: rocket volley 100%->150%; 14-part upgrade now also adds 30% damage]',
                                    '[Patch 13.04: added 14-part upgrade] [Balance 5-16-26: missile dmg 100%->75%; 1s shot cooldown; 14-part bonus 30%->15%]')
            changed = True

    # METEOR
    elif uid == 'meteor':
        if talent == 'Fiery Bolide':
            row[7] = desc.replace('Maximum 700 charges.',
                                  'Maximum 900 charges (was 700). For charges exceeding the maximum, each additional charge increases Meteor damage by +1% and Meteorite damage by +0.5%.')
            row[7] += ' [Balance 5-16-26: max charges 700->900; over-cap bonus mechanic added]'
            changed = True
        elif talent == 'Apocalypse':
            row[7] = desc.replace('has a 20% chance to summon', 'has a 25% chance to summon (was 20%)')
            row[7] += ' [Balance 5-16-26: summon chance 20%->25%]'
            changed = True

    if changed:
        row[17] = '2026-05-21'
        modified += 1

print(f"Rows modified: {modified}")

# Insert Dark Moloch talent rows after the last cultist talent row
cultist_last_idx = max(i for i, r in enumerate(rows) if r and r[0].strip() == 'cultist')
print(f"Last cultist talent row index: {cultist_last_idx}")

dark_moloch_talents = [
    ['dark_moloch','Dark Moloch','','1','L','9','Legion',
     "The Overlords' damage is increased by 40. Receiving a Mark has an equal chance of summoning a Moloch of merge rank 1 or 2.",
     '','','10','','','','Yes','Complete','In-game screenshot','2026-05-21'],
    ['dark_moloch','Dark Moloch','','1','R','9','Emperor',
     ("The first Moloch on the battlefield becomes the Emperor who can increase his merge rank only by absorbing a unit of the same rank. "
      "The Emperor's attack speed is increased by 10%, and the Mark limit is increased by 150%. When a Mark is received, a random enemy unit "
      "is stunned for 3 sec, and loses a merge rank. A Moloch of merge rank 4 releases the same type of projectile as other Molochs and Overlords. "
      "[Balance 5-16-26: fixed stun ignoring shields]"),
     '','','10','','','','Yes','Complete','In-game screenshot','2026-05-21'],
    ['dark_moloch','Dark Moloch','','2','L','11','Cosmic Protection',
     'When ranking up, a Moloch has a 30% chance to be cleansed of negative effects.',
     '','','12','','','','Yes','Complete','In-game screenshot','2026-05-21'],
    ['dark_moloch','Dark Moloch','','2','R','11','Mark of Growth',
     'Upon receiving a mark, a Moloch has a 15% chance to increase the merge rank twice.',
     '','','12','','','','Yes','Complete','In-game screenshot','2026-05-21'],
    ['dark_moloch','Dark Moloch','','3','L','13','Infinite Power',
     "The Overlords and the Emperor start attacking with lightning, dealing 120% of the Moloch's damage to 5 enemies in a chain.",
     '','','14','','','','Yes','Complete','In-game screenshot','2026-05-21'],
    ['dark_moloch','Dark Moloch','','3','R','13','Molten Armor',
     "A Moloch's attacks have a 5% chance to reduce a monster's armor by 30%.",
     '','','14','','','','Yes','Complete','In-game screenshot','2026-05-21'],
    ['dark_moloch','Dark Moloch','','4','Fixed','15','Marks of Infinity',
     "The Moloch's critical attack chance is increased by 0.6% per mark (up to a maximum of 6%). The damage bonus per mark after the limit is increased by 100%.",
     '','','15','','','','No','Complete','In-game screenshot','2026-05-21'],
]

insert_at = cultist_last_idx + 1
for j, row in enumerate(dark_moloch_talents):
    rows.insert(insert_at + j, row)

print(f"Inserted 7 Dark Moloch talent rows at index {insert_at}")

# Write back
out = io.StringIO()
writer = csv.writer(out, lineterminator='\r\n')
writer.writerows(rows)

with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(out.getvalue())

print("Talent Trees.csv updated successfully.")
print(f"Total rows now: {len(rows)}")